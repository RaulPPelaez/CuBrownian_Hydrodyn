all: 
	$(MAKE) -C src/ all
	mv src/main bin/brow
clean:
	rm bin/brow
	$(MAKE) -C src/ clean
redo:
	$(MAKE) -C src/ redo 
	mv src/main bin/brow
