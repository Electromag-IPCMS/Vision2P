import os
import sys
sys.path.insert(0, os.path.abspath('../'))

project = 'Vision2P'
author = 'CNRS, CroesBoris, Cherifi-Hertel Salia'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
]

source_suffix = '.rst'
master_doc = 'index' 
exclude_patterns = []
html_show_sphinx = False
html_theme = 'sphinx_rtd_theme'



html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
}

htmlhelp_basename = 'Vision2Pdoc'

latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
}
