# Out-of-kit build for the vendored inscount tool. The kit's own
# makefile does the heavy lifting; PIN_ROOT must point at the
# extracted kit root (the directory containing the `pin` launcher).
ifdef PIN_ROOT
CONFIG_ROOT := $(PIN_ROOT)/source/tools/Config
else
CONFIG_ROOT := ../Config
endif
include $(CONFIG_ROOT)/makefile.config
include makefile.rules
include $(TOOLS_ROOT)/Config/makefile.default.rules
