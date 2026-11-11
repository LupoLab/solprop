// SPDX-FileCopyrightText: 2026 John C. Travers and contributors
//
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

/*
 * The boundary monitor's transverse edge, owned by a native session (solprop SEP 40
 * section 5.3, SEP 47 sections 4.3, 5.3 and 6.4, and the maintainer's decision 4 at its
 * acceptance).
 *
 * `Edge` is solprop's `TransverseEdge` less its reductions: the launch and its
 * judgements, the read spacing and its ceiling on the step, each read's ledgers and
 * judgement, a requested collar's current and return, the record a resume restores and
 * the record it keeps. Its every decision state and record are here; solprop formats
 * and emits what it reports and computes nothing of it.
 *
 * Its scalar decisions and the record's scalar arithmetic are strict (this translation
 * unit takes no contraction) and in solprop's order, with Python's comparison
 * semantics, so on identical readings they are solprop's exactly. That includes the
 * reach's tail sums, which NumPy accumulates in order (`np.cumsum`) over the columns
 * sorted as solprop sorted them (the order arrives with the configuration, so no tie is
 * broken here), and the worst face, whose ties `max` breaks on the face's name.
 *
 * Three reductions belong to the reading instead (SEP 47 section 4.3) and are compared
 * under SEP 25: the launch's energy in the safety ramp's band and its total, and the
 * ramp's removal over a read interval, `sum_c P_c (1 - exp(-alpha_c h))`. NumPy sums
 * them pairwise and its `expm1` may be its own; here they are summed in order with the
 * C library's. A caller holding solprop's own values hands them over (`LaunchSums`,
 * `ReadSums`), as the recorded-readings harness does; a session lets the edge form
 * them. The input noise's readings (`Noise`) are always the caller's.
 *
 * A launch or a read can raise several warnings at one point, each of which solprop
 * may turn into an exception under the caller's filters. Each call therefore stops at
 * its first notice, keeping where it was, and `proceed` does the rest of that point's
 * work up to the next notice: a session makes each warning its own service barrier.
 *
 * Not thread-safe: the session's lock serialises it.
 */
namespace solprop_native::transverse {

/** The run's transverse status (`TransverseStatus`). */
enum class Status : int {
  clear = 0,
  warned = 1,
  stopped = 2,
  insensitive = 3,
  coarse = 4,
  unwatched = 5,
};

/** Why the status is not a verdict (`_reason`): none, an unread edge, or the input
    noise's allowance, whose text solprop formats from the record. */
enum class Reason : int { none = 0, unwatched = 1, insensitive = 2 };

/** Every number the edge reads, formed by solprop from its `TransverseEdge` and module
    constants. */
struct Config {
  /** The faces' names, in solprop's order: a tie between two faces' judged shares is
      broken on them, as `max` breaks a tuple's. One, two or four. */
  std::vector<std::string> faces;
  /** The warning and stop as energy shares: `budget ** 2` and
      `(stop_factor * budget) ** 2`. */
  double warn = 0.0;
  double stop = 0.0;
  /** The guide's length, m; the narrowest face's guard width, m (`_guard`); the
      smallest wavenumber over the band, rad/m; and `RAMP_K_FLOOR`. */
  double length = 0.0;
  double guard = 0.0;
  double k_min = 0.0;
  double k_floor = 0.0;
  /** `READS`, `BACKSTOP_READS`, `REACH_GROWTH` and `NOISE_FACTOR`. */
  double reads = 16.0;
  double backstop_reads = 4.0;
  double reach_growth = 0.1;
  double noise_factor = 8.0;
  /** Each spectral column's weight (`_columns`), its transverse wavenumber (`_kappa`)
      and the columns by increasing wavenumber, as solprop sorted them
      (`_kappa_order`). */
  std::vector<double> columns;
  std::vector<double> kappa;
  std::vector<std::size_t> kappa_order;
  /** A transverse boundary (`design is not None`), with its safety ramp's power rate
      on each column, 1/m; a requested collar (`design.has_collar`), read at every
      committed step; and whether it has a rate (`design.collar_rate is not None`). */
  bool design = false;
  std::vector<double> ramp_rate;
  bool collar = false;
  bool collar_rate = false;
  /** Whether content above `discard_above` was declared; whether the backend reads
      the edge (every native session does). */
  bool declared = false;
  bool reads_edge = true;
  /** A fixed schedule's longest step, m, which the spacing cannot limit; none
      without a schedule. */
  std::optional<double> longest_step;
  /** The input noise's allowance takes the first face's share of the weights,
      `sum(w[edge]) / sum(w)`, and its count of modes, `max(nodes, 1) max(band, 1)`. */
  double noise_share = 0.0;
  double noise_modes = 1.0;
};

/** What a backend reads of its state for the edge (`_EdgeReading`): numbers, never the
    field. Each part is present only when it was asked for. */
struct Reading {
  /** The faces' region sums (`faces`), and each column's unweighted energy; both or
      neither. */
  std::optional<std::array<double, 4>> faces;
  std::vector<double> columns;
  /** The launch's clip sums and the totals: the grid, a collar's rate, outside the
      request. */
  std::optional<std::array<double, 4>> clip;
  std::optional<std::array<double, 4>> totals;
  /** A requested collar's `(inward, windowed)` per line. */
  std::optional<std::vector<double>> strip;
  /** A requested collar's rectified inward current. */
  std::optional<double> current;
};

/** The input noise a launch carries, as readings of the noise draw. */
struct Noise {
  /** Each column's unweighted energy (`column_power(noise)`). */
  std::vector<double> columns;
  /** Its node energy, `sum(w node_power(noise))`. */
  double energy = 0.0;
  /** Its weighted columns' energy in the ramp's band, `sum(noise_columns[ramp])`. */
  double ramp = 0.0;
  /** A requested collar's strip of it, `(inward, windowed)` per line. */
  std::vector<double> strip;
  /** The launch's columns without it, unweighted, for the ramp's band; none to judge
      the launch's own. */
  std::optional<std::vector<double>> clean;
};

/** The launch's ramp-band reading: the judged columns' total and their energy in the
    band (`_judge_ramp_band`). */
struct LaunchSums {
  double total = 0.0;
  double ramp = 0.0;
};

/** A read interval's ramp reading: `sum(last_columns taken)` and
    `sum(noise_columns taken)`, `taken = -expm1(-ramp_rate gap)`. */
struct ReadSums {
  double raw = 0.0;
  double noise = 0.0;
};

/** One verdict a call reports: a warning, or a stop when `stop` is set. solprop
    formats its message from these numbers alone. */
struct Notice {
  enum class Kind : int {
    none = 0,
    ramp_band = 1,    ///< the launch's share in the safety ramp's band (`value`)
    unwatched = 2,    ///< the backend does not read the edge
    clip = 3,         ///< a face's clip of the launch (`face`, `value`)
    insensitive = 4,  ///< the input noise's allowance (`value`)
    coarse = 5,       ///< a fixed schedule's longest step (`value`) past `spacing`
    face = 6,         ///< a face's judged share (`face`, `value`) at `z`; a stop names
                      ///< where it first warned, `warned_at` (NaN for nowhere)
    returned = 7,     ///< a collar's returned share (`value`) at `z`
    ramp_removal = 8, ///< the safety ramp's removal (`value`)
  };
  Kind kind = Kind::none;
  bool stop = false;
  std::size_t face = 0;
  double value = 0.0;
  double z = 0.0;
  double spacing = 0.0;
  double warned_at = std::numeric_limits<double>::quiet_NaN();
};

/** An earlier execution's `transverse` record as a resume reads it
    (`TransverseEdge.resume`): each number present in it and the per-face lists, the
    reads already whole. solprop reads the record's types; which fields are restored,
    their defaults and the test that the record fits are made here. */
struct Resumption {
  /** Whether there is a record at all. */
  bool record = false;
  std::optional<double> launch_energy;
  std::optional<std::vector<double>> edge_baseline;
  std::optional<std::vector<double>> edge_max;
  std::optional<std::vector<double>> edge_max_raw;
  std::optional<double> noise_allowance;
  std::optional<double> raw_ramp_removed;
  std::optional<double> noise_ramp_removed;
  std::optional<double> noise_share_in_ramp;
  std::optional<double> declared_share_in_ramp;
  std::optional<double> spatial_removed;
  std::optional<double> current;
  std::optional<double> current_baseline;
  std::optional<double> strip_baseline;
  std::optional<double> strip_floor;
  std::optional<double> strip_max;
  std::optional<double> returned_max;
  /** `int(number("reads"))`, which solprop forms. */
  std::int64_t reads = 0;
  std::optional<double> largest_gap;
  std::optional<double> warned_at;
  /** The record's `transverse_status`, when it is a string. */
  std::optional<std::string> status;
};

/** The edge's record (`TransverseEdge.summary`'s numbers): solprop adds the faces'
    names, the reason's text and `None` where a number was never measured. */
struct Record {
  double launch_energy = std::numeric_limits<double>::quiet_NaN();
  double noise_ramp_removed = 0.0;
  double noise_share_in_ramp = 0.0;
  double declared_share_in_ramp = 0.0;
  Status status = Status::clear;
  Reason reason = Reason::none;
  std::vector<double> edge_baseline;
  std::vector<double> edge_max;
  std::vector<double> edge_max_raw;
  double noise_allowance = 0.0;
  double ramp_removed = 0.0;
  double raw_ramp_removed = 0.0;
  double spatial_removed = 0.0;
  std::int64_t reads = 0;
  double returned_max = 0.0;
  double strip_max = 0.0;
  double strip_baseline = 0.0;
  double strip_floor = 0.0;
  double current = 0.0;
  double current_baseline = 0.0;
  double largest_gap = 0.0;
  double warned_at = std::numeric_limits<double>::quiet_NaN();
  /** Whether the edge continues an earlier record (`resumed`), not a summary field. */
  bool resumed = false;
};

/** The end of a run (`TransverseEdge.finish`'s log line): the status, the largest
    judged share (`max(edge_max, default=0.0)`), the ramp's removal and the reads. */
struct Finish {
  Status status = Status::clear;
  double largest = 0.0;
  double ramp_removed = 0.0;
  std::int64_t reads = 0;
};

/** `TransverseEdge`, owned natively. */
class Edge {
public:
  /** Validate `config`; throw `std::invalid_argument` naming the field at fault. */
  explicit Edge(Config config);

  /** `resume`: continue an earlier execution's record, before `launch`. */
  void resume(const Resumption& record);
  /** `resumed`. */
  [[nodiscard]] bool resumed() const noexcept { return resumed_; }
  /** `launch`: record the launched state's reading at `z`, judge it unless resumed,
      and set the first read; up to its first notice. `sums` are solprop's ramp-band
      reading, formed here when null; `noise` the input noise's readings, or null. */
  Notice launch(const Reading& reading, double z, const LaunchSums* sums = nullptr,
                const Noise* noise = nullptr);
  /** `read`: take a reading at a committed position, credit the removals and judge;
      up to its first notice. `sums` are solprop's ramp reading over the interval,
      formed here when null. A read the edge does not want records nothing. */
  Notice read(const Reading& reading, double z, const ReadSums* sums = nullptr);
  /** The rest of the point whose last call stopped at a warning, up to its next
      notice; none when nothing is left. */
  Notice proceed();
  /** Whether a warning left the rest of its point to do. */
  [[nodiscard]] bool pending() const noexcept { return stage_ != Stage::none; }
  /** `step`: accumulate a collar's inward current at a committed step. */
  void step(double current, double z);

  /** `ceiling`: the longest step the reads allow, m; infinite when nothing is read. */
  [[nodiscard]] double ceiling() const noexcept;
  /** `due`: whether the edge is read at the committed position `z`, the next step
      being `proposal`. */
  [[nodiscard]] bool due(double z, double proposal) const;
  /** `wants_read`. */
  [[nodiscard]] bool wants_read(double z) const;
  /** `next_read`, m. */
  [[nodiscard]] double next_read() const noexcept { return next_; }
  /** `every_step`: a requested collar, read at every committed step. */
  [[nodiscard]] bool every_step() const noexcept {
    return config_.design && config_.collar;
  }
  /** `ramp_removed`. */
  [[nodiscard]] double ramp_removed() const;
  /** `status`. */
  [[nodiscard]] Status status() const noexcept { return status_; }
  /** `summary`, its numbers. */
  [[nodiscard]] Record record() const;
  /** `finish`, its numbers. */
  [[nodiscard]] Finish finish() const;
  [[nodiscard]] const Config& config() const noexcept { return config_; }

  /** The launch's ramp-band reading from `columns` (unweighted), summed in order. */
  [[nodiscard]] LaunchSums launch_sums(const std::vector<double>& columns) const;
  /** The ramp reading over a read interval of `gap`, m, from the columns last read,
      summed in order. */
  [[nodiscard]] ReadSums read_sums(double gap) const;

private:
  enum class Stage : int {
    none = 0,
    // The launch, after its ramp band's warning, before the unread edge's verdict.
    launch_unread = 1,
    // The launch's clip, from face `face_`.
    launch_clip = 2,
    launch_noise = 3,
    launch_spacing = 4,
    launch_rest = 5,
    // A read, after its face or return judgement: the ramp's removal.
    read_ramp = 6,
  };

  Notice run_launch();
  [[nodiscard]] std::vector<double> weighted(const std::vector<double>& columns) const;
  [[nodiscard]] double collar_rate(const Reading& reading) const;
  [[nodiscard]] double reach(const std::vector<double>& columns) const;
  void respace(const std::vector<double>& columns, bool launch);
  [[nodiscard]] std::array<double, 2>
  strip_shares(const std::vector<double>& strip) const;
  Notice judge_ramp_band(const LaunchSums& sums);
  void launch_return();
  Notice judge_faces(const Reading& reading, double z);
  Notice judge_return(const Reading& reading, double z);
  Notice judge_ramp_removal();

  Config config_;
  std::size_t n_faces_ = 0;
  double launch_ = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> baseline_;
  std::vector<double> peak_;
  std::vector<double> peak_raw_;
  double allowance_ = 0.0;
  bool has_noise_ = false;
  std::vector<double> noise_columns_;
  double noise_share_in_ramp_ = 0.0;
  double ramp_raw_ = 0.0;
  double ramp_noise_ = 0.0;
  double spatial_removed_ = 0.0;
  bool has_last_rate_ = false;
  double last_rate_z_ = 0.0;
  double last_rate_ = 0.0;
  bool has_last_columns_ = false;
  std::vector<double> last_columns_;
  double last_z_ = std::numeric_limits<double>::quiet_NaN();
  double next_ = std::numeric_limits<double>::infinity();
  double spacing_now_ = std::numeric_limits<double>::infinity();
  double backstop_ = std::numeric_limits<double>::infinity();
  double reach_reference_ = 0.0;
  std::int64_t reads_ = 0;
  double largest_gap_ = 0.0;
  double warned_at_ = std::numeric_limits<double>::quiet_NaN();
  Status status_ = Status::clear;
  Reason reason_ = Reason::none;
  bool warned_ = false;
  bool ramp_warned_ = false;
  bool resumed_ = false;
  double declared_share_ = 0.0;
  double strip_baseline_ = 0.0;
  double strip_floor_ = 0.0;
  double strip_peak_ = 0.0;
  double current_ = 0.0;
  bool has_current_last_ = false;
  double current_last_z_ = 0.0;
  double current_last_ = 0.0;
  double current_baseline_ = 0.0;
  double returned_peak_ = 0.0;

  // Where a call that stopped at a warning left its point, and what the rest reads.
  Stage stage_ = Stage::none;
  std::size_t face_ = 0;
  Reading held_;
  double held_z_ = 0.0;
  std::optional<Noise> held_noise_;
};

/** The strictness test's witness, as this translation unit rounds `a b + c`; never
    inlined, as `march::strict_probe`. */
[[nodiscard, gnu::noinline]] double strict_probe(double a, double b, double c);

} // namespace solprop_native::transverse
