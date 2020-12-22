all:
	$(MAKE) -C src/lab1/
	$(MAKE) -C src/lab2/
	$(MAKE) -C src/lab3/

clean:
	$(MAKE) clean -C src/lab1/
	$(MAKE) clean -C src/lab2/
	$(MAKE) clean -C src/lab3/
