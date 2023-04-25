all: apack_extension_module

apack_extension_module:
	mkdir -p build/
	cd build/ && python ../src/setup.py install

clean:
	rm -rf build/
	rm -rf env/lib/python3.9/site-packages/*.egg