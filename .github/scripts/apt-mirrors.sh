#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

set -euo pipefail

# The runner's Azure mirror repeatedly failed DNS resolution before our gates (#757).
# Keep its official Ubuntu archive/security fallbacks and the existing signed sources.
# Runner configuration: actions/runner-images, images/ubuntu/scripts/build/configure-apt-sources.sh
mirrors=/etc/apt/apt-mirrors.txt
if [[ -f "$mirrors" ]]; then
    sudo sed -i '/azure\.archive\.ubuntu\.com/d' "$mirrors"
fi
