# Simple Makefile which just runs through all the ALL targets
# and does BMC and prove on them.

ALL1 := op op_imm lui auipc
ALL2 := jal jalr branch csr
ALL3 := lb lbu lh lhu lw sb sh sw fatal
BMC1 := $(patsubst %,%-bmc,$(ALL1))
BMC2 := $(patsubst %,%-bmc,$(ALL2))
BMC3 := $(patsubst %,%-bmc,$(ALL3))
PROVE1 := $(patsubst %,%-prove,$(ALL1))
PROVE2 := $(patsubst %,%-prove,$(ALL2))
PROVE3 := $(patsubst %,%-prove,$(ALL3))

# Here we specify the longest-running targets first, so they get priority
# when parallelizing with make -j. In theory, if there's one of those long-running
# targets which takes the longest time, it will have started approximately first.
ALLBMC := $(BMC3) $(BMC2) $(BMC1) 
ALLPROVE := $(PROVE3) $(PROVE2) $(PROVE1)
ALL := $(ALLBMC) $(ALLPROVE)

SRCS := formal_cpu.py sequencer_card.py reg_card.py shift_card.py alu_card.py
SRCS += transparent_latch.py async_memory.py util.py consts.py

all: | $(ALLBMC)
	@for i in $(ALL1) $(ALL2) $(ALL3); do \
	  DIR="formal_cpu_$$i-bmc"; \
	  if [ -e "$$DIR" -a -f "$$DIR/status" ]; then \
	    printf "%-15s: %s\n" "$$i-bmc" "`cat $$DIR/status`"; \
	  else \
	  	printf "%-15s: %s\n" "$$i-bmc" "UNCOMPLETED"; \
	  fi; \
	done

clean: cleanbmc cleanprove

cover: $(SRCS)
	python3 formal_cpu.py gen
	sby -f formal_cpu.sby cover

cleanbmc:
	@for i in $(ALLBMC); do \
	  rm -f formal_cpu_$(patsubst %-bmc,%,$$i).il; \
	  rm -rf formal_cpu_$$i; \
	done

cleanprove:
	@for i in $(ALLPROVE); do \
	  rm -f formal_cpu_$(patsubst %-prove,%,$$i).il; \
	  rm -rf formal_cpu_$$i; \
	done

%-prove: formal_cpu_%.il
	sby -f formal_cpu.sby $@

formal_cpu_%.il: VERIFY = $(patsubst formal_cpu_%.il,%,$@)
formal_cpu_%.il: $(SRCS)
	python3 formal_cpu.py gen $(VERIFY)

%-bmc: %-bmc/.done
	printf "\n"

%-bmc/.done: VERIFY = $(patsubst %-bmc/.done,%,$@)
%-bmc/.done: formal_cpu_%.il
	sby -f formal_cpu.sby $(VERIFY)-bmc
	touch formal_cpu_$(VERIFY)-bmc/.done
