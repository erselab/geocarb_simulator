geocarb_simulator
==================

A whole-slit GeoCarb CO2 retrieval simulator built on top of `gert
<https://github.com/erselab/gert>`_ (the shared radiative-transfer and
optimal-estimation library) -- geocarb_simulator supplies GeoCarb's own
instrument/scan geometry, the along-slit truth scene, and the joint-block
retrieval driver; ``gert`` supplies the radiative transfer and the core
Gauss-Newton machinery.

``docs/PROJECT_STATUS.md`` is this project's own dated decision log --
every non-obvious design choice, bug, and validation result, in the order
it happened. Start there for *why* something is the way it is; the API
reference below is for *what* is actually callable.

.. toctree::
   :maxdepth: 1
   :caption: Project

   PROJECT_STATUS.md
   ALGORITHM_ROADMAP.md
   readme.md

.. toctree::
   :maxdepth: 2
   :caption: API reference — geocarb_gert

   api/geocarb_gert
   api/geocarb_gert.adapter
   api/geocarb_gert.aerosol_defaults
   api/geocarb_gert.along_slit_query
   api/geocarb_gert.along_slit_scene
   api/geocarb_gert.along_slit_state
   api/geocarb_gert.cross_band
   api/geocarb_gert.focalplane
   api/geocarb_gert.gd_polynomials
   api/geocarb_gert.gd_render
   api/geocarb_gert.instrument
   api/geocarb_gert.jacobians
   api/geocarb_gert.joint_state
   api/geocarb_gert.levels
   api/geocarb_gert.mission_config
   api/geocarb_gert.paths
   api/geocarb_gert.radiometry
   api/geocarb_gert.robust_stats
   api/geocarb_gert.scene
   api/geocarb_gert.spectrum
   api/geocarb_gert.truth_cache

.. toctree::
   :maxdepth: 1
   :caption: API reference — driver & validation scripts

   api/scripts.gd_joint_block_retrieve
   api/scripts.gd_jacobian_validate
   api/scripts.gd_xrtm_aerosol_jacobian_validate

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
