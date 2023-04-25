SUBDIRS = old_baseline_cpu # Other directories excluded for now...

.PHONY: subdirs $(SUBDIRS) clean

subdirs: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

clean:
	for dir in $(SUBDIRS); do \
    	$(MAKE) -C $$dir clean; \
	done
