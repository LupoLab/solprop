{{ fullname | escape | underline}}

{# A package re-exports the public names of its submodules, and each of those is also
   documented on the submodule's own page. Documenting them here too would register
   two objects under one name -- `solprop.physdata.density` is both a module and the
   function that module exports -- which Sphinx reports as a duplicate object
   description and `-W` turns into a build failure. So a package page carries only its
   module list; a leaf module documents its members as usual. #}
{%- if modules %}
.. automodule:: {{ fullname }}
   :no-members:

.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}

{#- The session namespace re-exports names that are documented on their home modules'
   pages, for the same reason a package page carries no members; only the four
   functions it defines itself are documented here. #}
{%- elif fullname == "solprop.sol" %}
.. automodule:: {{ fullname }}
   :no-members:

.. autofunction:: {{ fullname }}.start_session

.. autofunction:: {{ fullname }}.end_session

.. autofunction:: {{ fullname }}.set_log_level

.. autofunction:: {{ fullname }}.material_warnings

{%- else %}
.. automodule:: {{ fullname }}

   {% block attributes %}
   {%- if attributes %}
   .. rubric:: {{ _('Module Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block functions %}
   {%- if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block classes %}
   {%- if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {%- block exceptions %}
   {%- if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}
{%- endif %}
