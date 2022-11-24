"""Setup package for building"""
import setuptools
import sys
import versioneer

sys.path.insert(0, ".")
print(f"sys : {sys}")
setuptools.setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
