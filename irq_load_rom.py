# Disable pylint's "your name is too short" warning.
# pylint: disable=C0103
# Disable protected access warnings
# pylint: disable=W0212
from nmigen import Signal, Module, Elaboratable
from nmigen.build import Platform

from util import all_true


class IrqLoadInstrROM(Elaboratable):
    """ROM for the interrupt/load_instr sequencer card state machine."""

    def __init__(self):
        # Inputs (4 bits)

        self.is_interrupted = Signal()
        self._instr_phase = Signal(2)
        self.enable_sequencer_rom = Signal()

        # Outputs (4 bits)

        self.load_trap = Signal()
        self.next_trap = Signal()
        self._load_instr = Signal(reset=1)
        self.mem_rd = Signal()

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of the interrupt/load_instr sequencer ROM."""
        m = Module()

        # Defaults
        m.d.comb += [
            self.load_trap.eq(0),
            self.next_trap.eq(0),
            self._load_instr.eq(0),
            self.mem_rd.eq(0),
        ]

        with m.If(self.enable_sequencer_rom):

            # 3 independent cases:
            #
            # 1. We were interrupted and the instruction is complete.
            # 2. We're at the beginning of an instruction, and not interrupted.
            # 3. Instruction handling
            #
            # Maybe we can handle the first two with just some OR gates
            # on the output signals.

            # True when instr_complete and ~trap and (mei or mti pending).
            with m.If(self.is_interrupted):
                # m.d.comb += self._pc_to_z.eq(1)  # Does this do anything?
                # m.d.comb += self._next_instr_phase.eq(0)
                m.d.comb += self.load_trap.eq(1)
                m.d.comb += self.next_trap.eq(1)

            # Load the instruction on instruction phase 0 unless we've been interrupted.
            with m.If(all_true(self._instr_phase == 0, ~self.is_interrupted)):
                m.d.comb += self._load_instr.eq(1)
                m.d.comb += self.mem_rd.eq(1)

        return m
