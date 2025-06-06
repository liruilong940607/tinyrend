import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'tinyrend'
copyright = '2025, Ruilong Li'
author = 'Ruilong Li'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'breathe'
]

# Breathe configuration
breathe_projects = {
    "tinyrend": "../../build/docs/doxygen/xml"
}
breathe_default_project = "tinyrend"

# Theme
html_theme = 'sphinx_rtd_theme'

# Source files
source_suffix = '.rst'
master_doc = 'index' 