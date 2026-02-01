from setuptools import find_packages, setup

package_name = 'path_following_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='gm.ta.tahmasebi@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'pid_path_follower = path_following_controller.pid_path_follower:main',
            'mpc_path_follower = path_following_controller.mpc_path_follower:main',
        ],
    },
)
