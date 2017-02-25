import os
if os.environ.has_key('MUSERHOME'):
    task_directory = os.environ['MUSERHOME']+'/python'
else:
    task_directory = "/Users/wangfeng/museros/python"
if os.environ.has_key('MUSERHOME'):
    python_library_directory = os.environ['MUSERHOME']+'/python'
else:
    python_library_directory = "/Users/wangfeng/museros/python"
muser_version = "1.0.0"
subversion_revision = "1"
subversion_date = "2016-2-8 13:01:35 0800"
subversion_url = "https://www.cnlab.net"
build_time = "Wed 2016/2/8 12:39:00 UTC"
