# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from setuptools import find_packages, setup

setup(
    name='gnnlens',
    version='0.1.0',
    description='Visualization toolkit for Graph Neural Network',
    zip_safe=False,
    maintainer='GNNLens Team',
    url='https://github.com/dmlc/GNNLens2',
    install_requires=[
        'click',
        'flask>=1.0',
        'flask-cors>=3.0.9',
        'sqlalchemy>=1.1.14',
        'flask_compress',
        'numpy'
    ],
    packages=find_packages(),
    package_data={'gnnlens': ['visbuild/*', 'visbuild/static/js/*', 'visbuild/static/css/*', 'visbuild/static/media/*']},
    entry_points={
        'console_scripts': [
            'gnnlens=gnnlens.server:start_server'
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
    ],
    license='APACHE'
)