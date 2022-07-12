import os
if os.environ.get('https_proxy'):

 del os.environ['https_proxy']