from setuptools import setup

setup(
    name='gnnvis',
    version='0.0.0',
    url='http://flask.pocoo.org/docs/tutorial/',
    install_requires=[
        'flask',
    ],
    packages=['mini_server'],
    package_data={'mini_server': ['visbuild/*', 'visbuild/static/js/*', 'visbuild/static/css/*', 'visbuild/static/media/*']},
    entry_points={
        'console_scripts': [
            'gnnvis=mini_server.server:start_server'
        ],
    },
)