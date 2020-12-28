# Disable pylint's "your name is too short" warning.
# pylint: disable=C0103
# Disable protected access warnings
# pylint: disable=W0212
from nmigen import Signal, Module, Elaboratable
from nmigen.build import Platform

from consts import AluOp
from consts import TrapCauseSelect
from util import all_true


class TrapROM(Elaboratable):
    """ROM for the trap sequencer card state machine."""

    def __init__(self):
        # Inputs (12 bits)

        self.is_interrupted = Signal()
        self.exception = Signal()
        self.fatal = Signal()
        self.instr_misalign = Signal()
        self.bad_instr = Signal()
        self.trap = Signal()
        self.mei_pend = Signal()
        self.mti_pend = Signal()
        self.vec_mode = Signal(2)
        self._instr_phase = Signal(2)

        # Outputs (32 bits)

        self.set_instr_complete = Signal()

        # Raised when the exception card should store trap data.
        self.save_trap_csrs = Signal()

        # CSR lines
        self.csr_to_x = Signal()

        self._next_instr_phase = Signal(2)

        # -> X
        self._trapcause_to_x = Signal()

        # -> Y
        self._pc_to_y = Signal()
        self._pc_plus_4_to_y = Signal()
        self._mtvec_30_to_y = Signal()

        # -> Z
        self.alu_op_to_z = Signal(AluOp)  # 4 bits
        self._pc_to_z = Signal()
        self._instr_to_z = Signal()

        # -> PC
        self._z_30_to_pc = Signal()

        # -> csr_num
        self._mcause_to_csr_num = Signal()

        # -> memaddr
        self._z_30_to_memaddr = Signal()

        # -> various CSRs
        self._trapcause_select = Signal(TrapCauseSelect)  # 4 bits
        self.clear_pend_mti = Signal()
        self.clear_pend_mei = Signal()

        self.enter_trap = Signal()
        self.exit_trap = Signal()

        # Signals for next registers
        self.load_trap = Signal()
        self.next_trap = Signal()
        self.load_exception = Signal()
        self.next_exception = Signal()
        self.next_fatal = Signal()

        self.enable_sequencer_rom = Signal()

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of the trap sequencer ROM."""
        m = Module()

        # Defaults
        m.d.comb += [
            self.set_instr_complete.eq(0),
            self.save_trap_csrs.eq(0),
            self.csr_to_x.eq(0),
            self._next_instr_phase.eq(0),
            self._trapcause_to_x.eq(0),
            self._pc_to_y.eq(0),
            self._pc_plus_4_to_y.eq(0),
            self._mtvec_30_to_y.eq(0),
            self.alu_op_to_z.eq(AluOp.NONE),
            self._pc_to_z.eq(0),
            self._instr_to_z.eq(0),
            self._z_30_to_pc.eq(0),
            self._mcause_to_csr_num.eq(0),
            self._z_30_to_memaddr.eq(0),
            self._trapcause_select.eq(TrapCauseSelect.NONE),
            self.enter_trap.eq(0),
            self.exit_trap.eq(0),
            self.clear_pend_mti.eq(0),
            self.clear_pend_mei.eq(0),
            self.enable_sequencer_rom.eq(0),
        ]

        m.d.comb += [
            self.load_trap.eq(0),
            self.next_trap.eq(0),
            self.load_exception.eq(0),
            self.next_exception.eq(0),
            self.next_fatal.eq(0),
        ]

        # 4 cases here:
        #
        # 1. It's a trap!
        # 2. It's not a trap, but PC is misaligned.
        # 3. It's not a trap, PC is aligned, but it's a bad instruction.
        # 4. None of the above (allow control by sequencer ROM).

        with m.If(self.trap):
            self.handle_trap(m)

        # True when pc[0:2] != 0 and ~trap.
        with m.Elif(self.instr_misalign):
            self.set_exception(
                m, TrapCauseSelect.EXC_INSTR_ADDR_MISALIGN, mtval=self._pc_to_z)

        with m.Elif(self.bad_instr):
            self.set_exception(
                m, TrapCauseSelect.EXC_ILLEGAL_INSTR, mtval=self._instr_to_z)

        with m.Else():
            m.d.comb += self.enable_sequencer_rom.eq(1)

        return m

    def set_exception(self, m: Module, exc: TrapCauseSelect, mtval: Signal, fatal: bool = True):
        m.d.comb += self.load_exception.eq(1)
        m.d.comb += self.next_exception.eq(1)
        m.d.comb += self.next_fatal.eq(1 if fatal else 0)

        m.d.comb += self._trapcause_select.eq(exc)
        m.d.comb += self._trapcause_to_x.eq(1)

        m.d.comb += mtval.eq(1)  # what goes to z

        if fatal:
            m.d.comb += self._pc_to_y.eq(1)
        else:
            m.d.comb += self._pc_plus_4_to_y.eq(1)

        # X -> MCAUSE, Y -> MEPC, Z -> MTVAL
        m.d.comb += self.save_trap_csrs.eq(1)
        m.d.comb += self.load_trap.eq(1)
        m.d.comb += self.next_trap.eq(1)
        m.d.comb += self._next_instr_phase.eq(0)

    def handle_trap(self, m: Module):
        """Adds trap handling logic.

        For fatals, we store the cause and then halt.
        """
        is_int = ~self.exception

        with m.If(self._instr_phase == 0):
            with m.If(self.fatal):
                m.d.comb += self._next_instr_phase.eq(0)  # hang.
            with m.Else():
                m.d.comb += self._next_instr_phase.eq(1)

            # If set_exception was called, we've already saved the trap CSRs.
            with m.If(is_int):
                with m.If(self.mei_pend):
                    m.d.comb += self._trapcause_select.eq(
                        TrapCauseSelect.INT_MACH_EXTERNAL)
                    m.d.comb += self._trapcause_to_x.eq(1)
                    m.d.comb += self.clear_pend_mei.eq(1)
                with m.Elif(self.mti_pend):
                    m.d.comb += self._trapcause_select.eq(
                        TrapCauseSelect.INT_MACH_TIMER)
                    m.d.comb += self._trapcause_to_x.eq(1)
                    m.d.comb += self.clear_pend_mti.eq(1)

                m.d.comb += self._pc_to_y.eq(1)

                # MTVAL should be zero for non-exceptions, but right now it's just random.
                # X -> MCAUSE, Y -> MEPC, Z -> MTVAL
                m.d.comb += self.save_trap_csrs.eq(1)

        with m.Else():
            m.d.comb += self._mtvec_30_to_y.eq(1)  # mtvec >> 2
            with m.If(all_true(is_int, self.vec_mode == 1)):
                m.d.comb += [
                    self._mcause_to_csr_num.eq(1),
                    self.csr_to_x.eq(1),
                ]

            m.d.comb += self.load_trap.eq(1)
            m.d.comb += self.next_trap.eq(0)
            m.d.comb += [
                self.alu_op_to_z.eq(AluOp.ADD),
                self._z_30_to_memaddr.eq(1),  # z << 2 -> memaddr, pc
                self._z_30_to_pc.eq(1),
                self.enter_trap.eq(1),
                self.set_instr_complete.eq(1),
            ]
