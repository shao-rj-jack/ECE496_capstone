# Recursive invokation of Make, adapted from:
# https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html

SUBDIRS = cuda pytorch

.PHONY: subdirs $(SUBDIRS) clean

subdirs: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

# Require CUDA sources to be built before the pytorch wrapper
pytorch: cuda

clean:
	rm -rf pytorch/env/lib/python3.0/site-packages/*conv*.egg
	for dir in $(SUBDIRS); do \
    	$(MAKE) -C $$dir clean; \
	done