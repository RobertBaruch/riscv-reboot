# Disable pylint's "your name is too short" warning.
# pylint: disable=C0103
# Disable protected access warnings
# pylint: disable=W0212
from typing import List, Tuple

from nmigen import Signal, Module, Elaboratable, signed, ClockSignal, ClockDomain, Repl
from nmigen.build import Platform
from nmigen.asserts import Assert, Assume, Cover, Stable, Past

from consts import AluOp, AluFunc, BranchCond, CSRAddr, MemAccessWidth
from consts import Opcode, OpcodeFormat, SystemFunc, TrapCause, PrivFunc, MStatus
from consts import MInterrupt
from consts import InstrReg
from transparent_latch import TransparentLatch
from util import main
from IC_7416244 import IC_mux32
from IC_7416374 import IC_reg32_with_mux
from IC_GAL import IC_GAL_imm_format_decoder


def all_true(*args):
    cond = 1
    for arg in args:
        cond &= arg.bool()
    return cond


class SequencerState:
    """Contains only the registers in the sequencer card.

    This is useful to take snapshots of the entire state.
    """

    def __init__(self, ext_init: bool = False):
        attrs = [] if not ext_init else [("uninitialized", "")]

        self._pc = Signal(32, attrs=attrs)
        self._instr_phase = Signal(2)
        # Not quite a register, but the output of a latch
        self._instr = Signal(32)
        self._stored_alu_eq = Signal()
        self._stored_alu_lt = Signal()
        self._stored_alu_ltu = Signal()

        self.memaddr = Signal(32)
        self.memdata_wr = Signal(32)

        self._tmp = Signal(32)
        self.reg_page = Signal()

        # Trap handling
        # Goes high when we are handling a trap condition,
        # but not yet in the trap routine.
        self.trap = Signal()

        # Raised when an exception occurs.
        self._exception = Signal()

        self._mtvec = Signal(32, attrs=attrs)
        self._mcause = Signal(32, attrs=attrs)
        self._mepc = Signal(32, attrs=attrs)
        self._mtval = Signal(32, attrs=attrs)
        # Starts with interrupts globally disabled
        self._mstatus = Signal(32, attrs=attrs)
        # Specific interrupts enable
        self._mie = Signal(32, attrs=attrs)
        # Interrupts pending
        self._mip = Signal(32)


class SequencerCard(Elaboratable):
    """Logic for the sequencer card.

    Control lines indicate to the other cards what to do now,
    if the control lines control combinatorial logic, or
    to do next, if the control lines control sequential logic.

    This module uses two system-wide clocks: ph1 and ph2. The phases look
    like this:

           ________          ________
    ph1  _|   RD   |___WR___|   RD   |___WR___|
         ___     ____     ____     ____     ___
    ph2     |___|    |___|    |___|    |___|

    There's also a clock ph2w which is just ph2 on the WR phase:

         ____________     _____________     ___
    ph2w             |___|             |___|
    """

    def __init__(self, ext_init: bool = False, chips: bool = False):
        self.chips = chips
        self.ext_init = ext_init

        self.state = SequencerState(ext_init)

        # A clock-based signal, high only at the end of a machine
        # cycle (i.e. phase 5, the very end of the write phase).
        self.mcycle_end = Signal()

        # Control signals.
        self.alu_eq = Signal()
        self.alu_lt = Signal()
        self.alu_ltu = Signal()

        self.x_reg = Signal(5)
        self.y_reg = Signal(5)
        self.z_reg = Signal(5)

        # Raised when the processor halts because of an exception.
        self.fatal = Signal()
        # Raised when an interrupt based on an external timer goes off.
        self.time_irq = Signal()
        # Raised when any other external interrupt goes off.
        self.ext_irq = Signal()
        # Raised on the last phase of an instruction.
        self.instr_complete = Signal()

        # CSR lines
        self.csr_num = Signal(CSRAddr)
        self.csr_to_x = Signal()
        self.z_to_csr = Signal()

        # Buses, bidirectional
        self.data_x_in = Signal(32)
        self.data_x_out = Signal(32)
        self.data_y_in = Signal(32)
        self.data_y_out = Signal(32)
        self.data_z_in = Signal(32)
        self.data_z_out = Signal(32)

        # Memory
        self.mem_rd = Signal(reset=1)
        self.mem_wr = Signal()
        # Bytes in memory word to write
        self.mem_wr_mask = Signal(4)

        # Memory bus, bidirectional
        self.memdata_rd = Signal(32)

        # Internals

        # This opens the instr transparent latch to memdata. The enable
        # (i.e. load_instr) on the latch is a register, so setting load_instr
        # now opens the transparent latch next.
        self._load_instr = Signal(reset=1)

        self._instr_latch = TransparentLatch(32)
        self._pc_plus_4 = Signal(32)
        self._next_instr_phase = Signal(len(self.state._instr_phase))
        self._is_last_instr_cycle = Signal()
        self._next_reg_page = Signal()

        # Instruction decoding
        self._opcode = Signal(7)
        self._rs1 = Signal(5)
        self._rs2 = Signal(5)
        self._rd = Signal(5)
        self._funct3 = Signal(3)
        self._funct7 = Signal(7)
        self._funct12 = Signal(12)
        self._alu_func = Signal(4)
        self._imm_format = Signal(OpcodeFormat)
        self._imm = Signal(32)

        self._x_reg_select = Signal(InstrReg)
        self._y_reg_select = Signal(InstrReg)
        self._z_reg_select = Signal(InstrReg)

        # -> X
        self.reg_to_x = Signal()
        self._pc_to_x = Signal()
        self._memdata_to_x = Signal()

        # -> Y
        self.reg_to_y = Signal()
        self._imm_to_y = Signal()
        self._shamt_to_y = Signal()

        # -> Z
        self._pc_plus_4_to_z = Signal()
        self._tmp_to_z = Signal()
        self.alu_op_to_z = Signal(AluOp)  # 4 bits

        # -> PC
        self._pc_plus_4_to_pc = Signal()
        self._z_to_pc = Signal()
        self._x_to_pc = Signal()
        self._memaddr_to_pc = Signal()
        self._memdata_to_pc = Signal()

        # -> tmp
        self._x_to_tmp = Signal()

        # -> csr_num
        self._funct12_to_csr_num = Signal()
        self._mepc_num_to_csr_num = Signal()

        # -> memaddr
        self._pc_plus_4_to_memaddr = Signal()
        self._z_to_memaddr = Signal()
        self._x_to_memaddr = Signal()
        self._memdata_to_memaddr = Signal()

        # -> memdata
        self._z_to_memdata = Signal()

        # memory load shamt
        self._shamt = Signal(5)

        # -> various CSRs
        self._trapcause_to_mcause = Signal()
        self._trapcause = Signal(TrapCause)
        self._pc_to_mepc = Signal()
        self._pc_plus_4_to_mepc = Signal()
        self._pc_to_mtval = Signal()
        self._instr_to_mtval = Signal()
        self._memaddr_to_mtval = Signal()
        self._memaddr_lsb_masked_to_mtval = Signal()
        self._z_to_mtval = Signal()
        self._pend_mti = Signal()
        self._pend_mei = Signal()
        self._clear_pend_mti = Signal()
        self._clear_pend_mei = Signal()

        self._enter_trap = Signal()
        self._exit_trap = Signal()

    def elaborate(self, _: Platform) -> Module:
        """Implements the logic of the sequencer card."""
        m = Module()

        # Instruction latch
        m.submodules += self._instr_latch

        # Defaults
        m.d.comb += [
            self._next_instr_phase.eq(0),
            self.reg_to_x.eq(0),
            self._pc_to_x.eq(0),
            self._memdata_to_x.eq(0),
            self.reg_to_y.eq(0),
            self._imm_to_y.eq(0),
            self._shamt_to_y.eq(0),
            self.alu_op_to_z.eq(AluOp.NONE),
            self._pc_plus_4_to_z.eq(0),
            self._tmp_to_z.eq(0),
            self._pc_plus_4_to_pc.eq(0),
            self._x_to_pc.eq(0),
            self._z_to_pc.eq(0),
            self._memaddr_to_pc.eq(0),
            self._memdata_to_pc.eq(0),
            self._x_to_tmp.eq(0),
            self._pc_plus_4_to_memaddr.eq(0),
            self._x_to_memaddr.eq(0),
            self._z_to_memaddr.eq(0),
            self._memdata_to_memaddr.eq(0),
            self._z_to_memdata.eq(0),
            self.mem_rd.eq(0),
            self._load_instr.eq(0),
            self.mem_wr.eq(0),
            self.mem_wr_mask.eq(0),
            self._shamt.eq(0),
            self.data_x_out.eq(0),
            self.data_y_out.eq(0),
            self.data_z_out.eq(0),
            self._is_last_instr_cycle.eq(0),
            self.instr_complete.eq(0),
            self.csr_to_x.eq(0),
            self.z_to_csr.eq(0),
            self._funct12_to_csr_num.eq(0),
            self._mepc_num_to_csr_num.eq(0),
            self.csr_num.eq(0),
            self._x_reg_select.eq(0),
            self._y_reg_select.eq(0),
            self._z_reg_select.eq(0),
            self._next_reg_page.eq(self.state.reg_page),
            self._trapcause_to_mcause.eq(0),
            self._pc_to_mepc.eq(0),
            self._pc_plus_4_to_mepc.eq(0),
            self._pc_to_mtval.eq(0),
            self._instr_to_mtval.eq(0),
            self._memaddr_to_mtval.eq(0),
            self._memaddr_lsb_masked_to_mtval.eq(0),
            self._z_to_mtval.eq(0),
            self._enter_trap.eq(0),
            self._exit_trap.eq(0),
            self._pend_mti.eq(0),
            self._pend_mei.eq(0),
            self._clear_pend_mti.eq(0),
            self._clear_pend_mei.eq(0),
        ]
        m.d.comb += self._pc_plus_4.eq(self.state._pc + 4)

        self.process(m)

        return m

    def process(self, m: Module):
        # Latch the time and ext irqs. This cues them up for handling
        # when we're about to load an instruction (but not in a trap routine,
        # and not if interrupts are disabled).
        #
        # These get reset once the trap handler runs.
        #
        # The time irq always has higher priority than the ext irq.
        #
        # TODO: Should we make this triggered on the interrupt signals
        # themselves?
        with m.If(~self.state.trap):
            with m.If(self.state._mstatus[MStatus.MIE]):

                with m.If(self.state._mie[MInterrupt.MTI]):
                    with m.If(~self.state._mip[MInterrupt.MTI]):
                        m.d.comb += self._pend_mti.eq(self.time_irq)
                with m.Else():
                    m.d.comb += self._clear_pend_mti.eq(1)

                with m.If(self.state._mie[MInterrupt.MEI]):
                    with m.If(~self.state._mip[MInterrupt.MEI]):
                        m.d.comb += self._pend_mei.eq(self.ext_irq)
                with m.Else():
                    m.d.comb += self._clear_pend_mei.eq(1)

            with m.Else():
                m.d.comb += self._clear_pend_mti.eq(1)
                m.d.comb += self._clear_pend_mei.eq(1)

        with m.If(self._is_last_instr_cycle):
            m.d.comb += self.instr_complete.eq(self.mcycle_end)
            with m.If(~self._x_to_pc & ~self._z_to_pc & ~self._memaddr_to_pc & ~self._memdata_to_pc):
                m.d.comb += self._pc_plus_4_to_pc.eq(1)
            with m.If(~self._x_to_memaddr & ~self._z_to_memaddr & ~self._memdata_to_memaddr):
                m.d.comb += self._pc_plus_4_to_memaddr.eq(1)

        is_interrupt_pending = Signal()
        m.d.comb += is_interrupt_pending.eq(
            (self.time_irq & self.state._mie[MInterrupt.MTI]) |
            self.state._mip[MInterrupt.MTI] |
            (self.ext_irq & self.state._mie[MInterrupt.MEI]) |
            self.state._mip[MInterrupt.MEI]
        )
        # We only check interrupts once we're about to load an instruction but we're not already
        # trapping an interrupt.
        is_interrupted = all_true(self.instr_complete,
                                  ~self.state.trap,
                                  is_interrupt_pending)

        # Because we don't support the C (compressed instructions)
        # extension, the PC must be 32-bit aligned.
        instr_misalign = all_true(self.state._pc[0:2] != 0, ~self.state.trap)
        with m.If(instr_misalign):
            self.set_exception(
                m, TrapCause.EXC_INSTR_ADDR_MISALIGN, mtval=self._pc_to_mtval)

        with m.Elif(is_interrupted):
            m.d.comb += self._pc_to_mtval.eq(1)
            m.d.comb += self._next_instr_phase.eq(0)
            m.d.ph1 += self.state.trap.eq(1)

        # Load the instruction on instruction phase 0
        with m.Elif(all_true(self.state._instr_phase == 0, ~self.state.trap)):
            m.d.comb += self._load_instr.eq(1)
            m.d.comb += self.mem_rd.eq(1)

        read_pulse = ClockSignal("ph1") & ~ClockSignal("ph2")
        latch_instr = Signal()
        m.d.comb += [
            latch_instr.eq(read_pulse & self._load_instr),
            self.state._instr.eq(self._instr_latch.data_out),
            self._instr_latch.data_in.eq(self.memdata_rd),
            self._instr_latch.n_oe.eq(0),
            self._instr_latch.le.eq(latch_instr),
        ]

        # Updates to registers
        m.d.ph1 += self.state._instr_phase.eq(self._next_instr_phase)
        m.d.ph1 += self.state.reg_page.eq(self._next_reg_page)
        m.d.ph1 += self.state._stored_alu_eq.eq(self.alu_eq)
        m.d.ph1 += self.state._stored_alu_lt.eq(self.alu_lt)
        m.d.ph1 += self.state._stored_alu_ltu.eq(self.alu_ltu)

        if self.chips:
            self.multiplex_to_pc_chips(m)
            self.multiplex_to_memaddr_chips(m)
            self.multiplex_to_memdata_chips(m)
            self.multiplex_to_tmp_chips(m)
            self.multiplex_to_x_chips(m)
            self.multiplex_to_y_chips(m)
            self.multiplex_to_z_chips(m)
            self.multiplex_to_csr_num_chips(m)
            self.multiplex_to_reg_nums_chips(m)
            self.multiplex_to_csrs_chips(m)
        else:
            self.multiplex_to_pc(m)
            self.multiplex_to_memaddr(m)
            self.multiplex_to_memdata(m)
            self.multiplex_to_tmp(m)
            self.multiplex_to_x(m)
            self.multiplex_to_y(m)
            self.multiplex_to_z(m)
            self.multiplex_to_csr_num(m)
            self.multiplex_to_reg_nums(m)
            self.multiplex_to_csrs(m)

        # Decode instruction
        m.d.comb += [
            self._opcode.eq(self.state._instr[:7]),
            self._rs1.eq(self.state._instr[15:20]),
            self._rs2.eq(self.state._instr[20:25]),
            self._rd.eq(self.state._instr[7:12]),
            self._funct3.eq(self.state._instr[12:15]),
            self._funct7.eq(self.state._instr[25:]),
            self._alu_func[:3].eq(self._funct3),
            self._alu_func[3].eq(self._funct7[5]),
            self._funct12.eq(self.state._instr[20:]),
        ]
        if (self.chips):
            self.decode_imm_chips(m)
        else:
            self.decode_imm(m)

        # Handle the trap.
        with m.If(self.state.trap):
            self.handle_trap(m)

        with m.Elif(~instr_misalign):
            with m.If(self.state._instr[:16] == 0):
                self.handle_illegal_instr(m)

            with m.Elif(self.state._instr == 0xFFFFFFFF):
                self.handle_illegal_instr(m)

            with m.Else():
                # Output control signals
                with m.Switch(self._opcode):
                    with m.Case(Opcode.LUI):
                        self.handle_lui(m)

                    with m.Case(Opcode.AUIPC):
                        self.handle_auipc(m)

                    with m.Case(Opcode.OP_IMM):
                        self.handle_op_imm(m)

                    with m.Case(Opcode.OP):
                        self.handle_op(m)

                    with m.Case(Opcode.JAL):
                        self.handle_jal(m)

                    with m.Case(Opcode.JALR):
                        self.handle_jalr(m)

                    with m.Case(Opcode.BRANCH):
                        self.handle_branch(m)

                    with m.Case(Opcode.LOAD):
                        self.handle_load(m)

                    with m.Case(Opcode.STORE):
                        self.handle_store(m)

                    with m.Case(Opcode.SYSTEM):
                        self.handle_system(m)

                    with m.Default():
                        self.handle_illegal_instr(m)

    def set_exception(self, m: Module, exc: TrapCause, mtval: Signal, fatal: bool = True):
        m.d.ph2 += self.state._exception.eq(1)

        m.d.comb += self._trapcause.eq(exc)
        m.d.comb += self._trapcause_to_mcause.eq(1)

        m.d.comb += mtval.eq(1)

        if fatal:
            m.d.comb += self._pc_to_mepc.eq(1)
        else:
            m.d.comb += self._pc_plus_4_to_mepc.eq(1)

        m.d.ph1 += self.state.trap.eq(1)
        m.d.comb += self._next_instr_phase.eq(0)

    def handle_illegal_instr(self, m: Module):
        self.set_exception(m, TrapCause.EXC_ILLEGAL_INSTR,
                           mtval=self._instr_to_mtval)

    def multiplex_to_reg(self, m: Module, clk, reg: Signal, sels: List[Signal], sigs: List[Signal]):
        """Sets up a multiplexer with a register.

        clk is the clock domain on which the register is clocked.

        reg is the register signal.

        sels is an array of Signals which select that input for the multiplexer (active high). If
        no select is active, then the register retains its value.

        sigs is an array of Signals which are the inputs to the multiplexer.
        """
        assert(len(sels) == len(sigs))

        muxreg = IC_reg32_with_mux(
            clk=clk, N=len(sels), ext_init=self.ext_init)
        m.submodules += muxreg
        m.d.comb += reg.eq(muxreg.q)
        for i in range(len(sels)):
            m.d.comb += muxreg.n_sel[i].eq(~sels[i])
            m.d.comb += muxreg.d[i].eq(sigs[i])

    def multiplex_to_csrs(self, m: Module):
        with m.If(self.z_to_csr & (self.csr_num == CSRAddr.MCAUSE)):
            m.d.ph2w += self.state._mcause.eq(self.data_z_in)
        with m.Elif(self._trapcause_to_mcause):
            m.d.ph2w += self.state._mcause.eq(self._trapcause)

        with m.If(self.z_to_csr & (self.csr_num == CSRAddr.MEPC)):
            m.d.ph2w += self.state._mepc.eq(self.data_z_in)
        with m.Elif(self._pc_to_mepc):
            m.d.ph2w += self.state._mepc.eq(self.state._pc)
        with m.Elif(self._pc_plus_4_to_mepc):
            m.d.ph2w += self.state._mepc.eq(self._pc_plus_4)

        with m.If(self.z_to_csr & (self.csr_num == CSRAddr.MTVEC)):
            m.d.ph2w += self.state._mtvec.eq(self.data_z_in)

        with m.If(self.z_to_csr & (self.csr_num == CSRAddr.MTVAL)):
            m.d.ph2w += self.state._mtval.eq(self.data_z_in)
        with m.Elif(self._pc_to_mtval):
            m.d.ph2w += self.state._mtval.eq(self.state._pc)
        with m.Elif(self._instr_to_mtval):
            m.d.ph2w += self.state._mtval.eq(self.state._instr)
        with m.Elif(self._memaddr_to_mtval):
            m.d.ph2w += self.state._mtval.eq(self.state.memaddr)
        with m.Elif(self._memaddr_lsb_masked_to_mtval):
            m.d.ph2w += self.state._mtval.eq(self.state.memaddr & 0xFFFFFFFE)
        with m.Elif(self._z_to_mtval):
            m.d.ph2w += self.state._mtval.eq(self.data_z_in)

        with m.If(self.z_to_csr & (self.csr_num == CSRAddr.MSTATUS)):
            m.d.ph2w += self.state._mstatus.eq(self.data_z_in)
        with m.Elif(self._enter_trap):
            m.d.ph2w += self.state._mstatus[MStatus.MPIE].eq(
                self.state._mstatus[MStatus.MIE])
            m.d.ph2w += self.state._mstatus[MStatus.MIE].eq(0)
        with m.Elif(self._exit_trap):
            m.d.ph2w += self.state._mstatus[MStatus.MIE].eq(
                self.state._mstatus[MStatus.MPIE])
            m.d.ph2w += self.state._mstatus[MStatus.MPIE].eq(1)

        with m.If(self.z_to_csr & (self.csr_num == CSRAddr.MIE)):
            m.d.ph2w += self.state._mie.eq(self.data_z_in)

        with m.If(self.z_to_csr & (self.csr_num == CSRAddr.MIP)):
            # Pending machine interrupts are not writable.
            m.d.ph2w += self.state._mip[0:3].eq(self.data_z_in[0:3])
            m.d.ph2w += self.state._mip[4:7].eq(self.data_z_in[4:7])
            m.d.ph2w += self.state._mip[8:11].eq(self.data_z_in[8:11])
            m.d.ph2w += self.state._mip[12:].eq(self.data_z_in[12:])
        with m.If(self._pend_mti):
            m.d.ph2w += self.state._mip[MInterrupt.MTI].eq(1)
        with m.Elif(self._clear_pend_mti):
            m.d.ph2w += self.state._mip[MInterrupt.MTI].eq(0)
        with m.If(self._pend_mei):
            m.d.ph2w += self.state._mip[MInterrupt.MEI].eq(1)
        with m.Elif(self._clear_pend_mei):
            m.d.ph2w += self.state._mip[MInterrupt.MEI].eq(0)

    def multiplex_to_csrs_chips(self, m: Module):
        self.multiplex_to_reg(m, clk=m.d.ph2w, reg=self.state._mcause,
                              sels=[
                                  self.z_to_csr & (
                                      self.csr_num == CSRAddr.MCAUSE),
                                  self._trapcause_to_mcause
                              ],
                              sigs=[self.data_z_in, self._trapcause])
        self.multiplex_to_reg(m, clk=m.d.ph2w, reg=self.state._mepc,
                              sels=[
                                  self.z_to_csr & (
                                      self.csr_num == CSRAddr.MEPC),
                                  self._pc_to_mepc,
                                  self._pc_plus_4_to_mepc
                              ],
                              sigs=[self.data_z_in, self.state._pc, self._pc_plus_4])
        self.multiplex_to_reg(m, clk=m.d.ph2w, reg=self.state._mtvec,
                              sels=[
                                  self.z_to_csr & (
                                      self.csr_num == CSRAddr.MTVEC)
                              ],
                              sigs=[self.data_z_in])
        self.multiplex_to_reg(m, clk=m.d.ph2w, reg=self.state._mtval,
                              sels=[
                                  (self.z_to_csr & (
                                      self.csr_num == CSRAddr.MTVAL)) | self._z_to_mtval,
                                  self._pc_to_mtval,
                                  self._instr_to_mtval,
                                  self._memaddr_to_mtval,
                                  self._memaddr_lsb_masked_to_mtval,
                              ],
                              sigs=[
                                  self.data_z_in,
                                  self.state._pc,
                                  self.state._instr,
                                  self.state.memaddr,
                                  self.state.memaddr & 0xFFFFFFFE,
                              ])

        enter_trap_mstatus = self.state._mstatus
        enter_trap_mstatus &= ~(1 << MStatus.MIE)  # clear MIE
        enter_trap_mstatus &= ~(1 << MStatus.MPIE)  # clear MPIE
        enter_trap_mstatus |= (
            self.state._mstatus[MStatus.MIE] << MStatus.MPIE)  # set MPIE

        exit_trap_mstatus = self.state._mstatus
        exit_trap_mstatus |= (1 << MStatus.MPIE)  # set MPIE
        exit_trap_mstatus &= ~(1 << MStatus.MIE)  # clear MIE
        exit_trap_mstatus |= (
            self.state._mstatus[MStatus.MPIE] << MStatus.MIE)  # set MIE

        self.multiplex_to_reg(m, clk=m.d.ph2w, reg=self.state._mstatus,
                              sels=[
                                  self.z_to_csr & (
                                      self.csr_num == CSRAddr.MSTATUS),
                                  self._enter_trap,
                                  self._exit_trap,
                              ],
                              sigs=[
                                  self.data_z_in,
                                  enter_trap_mstatus,
                                  exit_trap_mstatus,
                              ])
        self.multiplex_to_reg(m, clk=m.d.ph2w, reg=self.state._mie,
                              sels=[
                                  self.z_to_csr & (
                                      self.csr_num == CSRAddr.MIE)
                              ],
                              sigs=[self.data_z_in])

        # Pending machine interrupts are not writable.
        mip_load = Signal(32)
        m.d.comb += mip_load.eq(self.data_z_in)
        m.d.comb += mip_load[MInterrupt.MTI].eq(
            self.state._mip[MInterrupt.MTI])
        m.d.comb += mip_load[MInterrupt.MEI].eq(
            self.state._mip[MInterrupt.MEI])
        m.d.comb += mip_load[MInterrupt.MSI].eq(
            self.state._mip[MInterrupt.MSI])

        mip_pend = Signal(32)
        m.d.comb += mip_pend.eq(self.state._mip)
        with m.If(self._pend_mti):
            m.d.comb += mip_pend[MInterrupt.MTI].eq(1)
        with m.Elif(self._clear_pend_mti):
            m.d.comb += mip_pend[MInterrupt.MTI].eq(0)
        with m.If(self._pend_mei):
            m.d.comb += mip_pend[MInterrupt.MEI].eq(1)
        with m.Elif(self._clear_pend_mei):
            m.d.comb += mip_pend[MInterrupt.MEI].eq(0)
        any_mip_pend = (self._pend_mti | self._clear_pend_mti |
                        self._pend_mei | self._clear_pend_mei)

        self.multiplex_to_reg(m, clk=m.d.ph2w, reg=self.state._mip,
                              sels=[
                                  self.z_to_csr & (
                                      self.csr_num == CSRAddr.MIP),
                                  any_mip_pend,
                              ],
                              sigs=[mip_load, mip_pend])

    def multiplex_to_pc(self, m: Module):
        with m.If(self._pc_plus_4_to_pc):
            m.d.ph1 += self.state._pc.eq(self._pc_plus_4)
        with m.Elif(self._x_to_pc):
            m.d.ph1 += self.state._pc.eq(self.data_x_in)
        with m.Elif(self._z_to_pc):
            m.d.ph1 += self.state._pc.eq(self.data_z_in)
        with m.Elif(self._memaddr_to_pc):
            # This is the result of a JAL or JALR instruction.
            # See the comment on JAL for why this is okay to do.
            m.d.ph1 += self.state._pc[1:].eq(self.state.memaddr[1:])
            m.d.ph1 += self.state._pc[0].eq(0)
        with m.Elif(self._memdata_to_pc):
            m.d.ph1 += self.state._pc.eq(self.memdata_rd)

    def multiplex_to_pc_chips(self, m: Module):
        self.multiplex_to_reg(m, clk=m.d.ph1, reg=self.state._pc,
                              sels=[
                                  self._pc_plus_4_to_pc,
                                  self._x_to_pc,
                                  self._z_to_pc,
                                  self._memaddr_to_pc,
                                  self._memdata_to_pc,
                              ],
                              sigs=[
                                  self._pc_plus_4,
                                  self.data_x_in,
                                  self.data_z_in,
                                  self.state.memaddr & 0xFFFFFFFE,
                                  self.memdata_rd,
                              ])

    def multiplex_to_tmp(self, m: Module):
        with m.If(self._x_to_tmp):
            m.d.ph2w += self.state._tmp.eq(self.data_x_in)

    def multiplex_to_tmp_chips(self, m: Module):
        self.multiplex_to_reg(m, clk=m.d.ph2w, reg=self.state._tmp,
                              sels=[
                                  self._x_to_tmp,
                              ],
                              sigs=[
                                  self.data_x_in,
                              ])

    def multiplex_to_memdata(self, m: Module):
        with m.If(self._z_to_memdata):
            m.d.ph1 += self.state.memdata_wr.eq(self.data_z_in)

    def multiplex_to_memdata_chips(self, m: Module):
        self.multiplex_to_reg(m, clk=m.d.ph1, reg=self.state.memdata_wr,
                              sels=[
                                  self._z_to_memdata,
                              ],
                              sigs=[
                                  self.data_z_in,
                              ])

    def multiplex_to_memaddr(self, m: Module):
        with m.If(self._pc_plus_4_to_memaddr):
            m.d.ph1 += self.state.memaddr.eq(self._pc_plus_4)
        with m.Elif(self._x_to_memaddr):
            m.d.ph1 += self.state.memaddr.eq(self.data_x_in)
        with m.Elif(self._z_to_memaddr):
            m.d.ph1 += self.state.memaddr.eq(self.data_z_in)
        with m.Elif(self._memdata_to_memaddr):
            m.d.ph1 += self.state.memaddr.eq(self.memdata_rd)

    def multiplex_to_memaddr_chips(self, m: Module):
        self.multiplex_to_reg(m, clk=m.d.ph1, reg=self.state.memaddr,
                              sels=[
                                  self._pc_plus_4_to_memaddr,
                                  self._x_to_memaddr,
                                  self._z_to_memaddr,
                                  self._memdata_to_memaddr,
                              ],
                              sigs=[
                                  self._pc_plus_4,
                                  self.data_x_in,
                                  self.data_z_in,
                                  self.memdata_rd
                              ])

    def multiplex_to_x(self, m: Module):
        with m.If(self._pc_to_x):
            m.d.comb += self.data_x_out.eq(self.state._pc)
        with m.Elif(self._memdata_to_x):
            m.d.comb += self.data_x_out.eq(self.memdata_rd)
        with m.Elif(self.csr_to_x):
            with m.Switch(self.csr_num):
                with m.Case(CSRAddr.MCAUSE):
                    m.d.comb += self.data_x_out.eq(self.state._mcause)
                with m.Case(CSRAddr.MTVEC):
                    m.d.comb += self.data_x_out.eq(self.state._mtvec)
                with m.Case(CSRAddr.MEPC):
                    m.d.comb += self.data_x_out.eq(self.state._mepc)
                with m.Case(CSRAddr.MTVAL):
                    m.d.comb += self.data_x_out.eq(self.state._mtval)
                with m.Case(CSRAddr.MSTATUS):
                    m.d.comb += self.data_x_out.eq(self.state._mstatus)
                with m.Case(CSRAddr.MIE):
                    m.d.comb += self.data_x_out.eq(self.state._mie)
                with m.Case(CSRAddr.MIP):
                    m.d.comb += self.data_x_out.eq(self.state._mip)

    def multiplex_to_x_chips(self, m: Module):
        mux = IC_mux32(9)
        m.submodules += mux

        inputs = [self.state._pc, self.memdata_rd, self.state._mcause, self.state._mtvec,
                  self.state._mepc, self.state._mtval, self.state._mstatus, self.state._mie,
                  self.state._mip]
        selectors = [
            self._pc_to_x,
            self._memdata_to_x,
            self.csr_to_x & (self.csr_num == CSRAddr.MCAUSE),
            self.csr_to_x & (self.csr_num == CSRAddr.MTVEC),
            self.csr_to_x & (self.csr_num == CSRAddr.MEPC),
            self.csr_to_x & (self.csr_num == CSRAddr.MTVAL),
            self.csr_to_x & (self.csr_num == CSRAddr.MSTATUS),
            self.csr_to_x & (self.csr_num == CSRAddr.MIE),
            self.csr_to_x & (self.csr_num == CSRAddr.MIP)
        ]

        for i in range(len(inputs)):
            m.d.comb += mux.a[i].eq(inputs[i])
            m.d.comb += mux.n_sel[i].eq(~selectors[i])
        m.d.comb += self.data_x_out.eq(mux.y)

    def multiplex_to_y(self, m: Module):
        with m.If(self._imm_to_y):
            m.d.comb += self.data_y_out.eq(self._imm)
        with m.Elif(self._shamt_to_y):
            m.d.comb += self.data_y_out.eq(self._shamt)

    def multiplex_to_y_chips(self, m: Module):
        mux = IC_mux32(2)
        m.submodules += mux

        m.d.comb += self.data_y_out.eq(mux.y)

        m.d.comb += mux.n_sel[0].eq(~self._imm_to_y)
        m.d.comb += mux.a[0].eq(self._imm)

        m.d.comb += mux.n_sel[1].eq(~self._shamt_to_y)
        m.d.comb += mux.a[1].eq(self._shamt)

    def multiplex_to_z(self, m: Module):
        with m.If(self._pc_plus_4_to_z):
            m.d.comb += self.data_z_out.eq(self._pc_plus_4)
        with m.Elif(self._tmp_to_z):
            m.d.comb += self.data_z_out.eq(self.state._tmp)

    def multiplex_to_z_chips(self, m: Module):
        mux = IC_mux32(2)
        m.submodules += mux

        m.d.comb += self.data_z_out.eq(mux.y)

        m.d.comb += mux.n_sel[0].eq(~self._pc_plus_4_to_z)
        m.d.comb += mux.a[0].eq(self._pc_plus_4)

        m.d.comb += mux.n_sel[1].eq(~self._tmp_to_z)
        m.d.comb += mux.a[1].eq(self.state._tmp)

    def multiplex_to_csr_num(self, m: Module):
        with m.If(self._funct12_to_csr_num):
            m.d.comb += self.csr_num.eq(self._funct12)
        with m.Elif(self._mepc_num_to_csr_num):
            m.d.comb += self.csr_num.eq(CSRAddr.MEPC)

    def multiplex_to_csr_num_chips(self, m: Module):
        mux = IC_mux32(2)
        m.submodules += mux

        m.d.comb += self.csr_num.eq(mux.y)

        m.d.comb += mux.n_sel[0].eq(~self._funct12_to_csr_num)
        m.d.comb += mux.a[0].eq(self._funct12)

        m.d.comb += mux.n_sel[1].eq(~self._mepc_num_to_csr_num)
        m.d.comb += mux.a[1].eq(CSRAddr.MEPC)

    def multiplex_to_reg_nums(self, m: Module):
        with m.Switch(self._x_reg_select):
            with m.Case(InstrReg.ZERO):
                m.d.comb += self.x_reg.eq(0)
            with m.Case(InstrReg.RS1):
                m.d.comb += self.x_reg.eq(self._rs1)
            with m.Case(InstrReg.RS2):
                m.d.comb += self.x_reg.eq(self._rs2)
            with m.Case(InstrReg.RD):
                m.d.comb += self.x_reg.eq(self._rd)
        with m.Switch(self._y_reg_select):
            with m.Case(InstrReg.ZERO):
                m.d.comb += self.y_reg.eq(0)
            with m.Case(InstrReg.RS1):
                m.d.comb += self.y_reg.eq(self._rs1)
            with m.Case(InstrReg.RS2):
                m.d.comb += self.y_reg.eq(self._rs2)
            with m.Case(InstrReg.RD):
                m.d.comb += self.y_reg.eq(self._rd)
        with m.Switch(self._z_reg_select):
            with m.Case(InstrReg.ZERO):
                m.d.comb += self.z_reg.eq(0)
            with m.Case(InstrReg.RS1):
                m.d.comb += self.z_reg.eq(self._rs1)
            with m.Case(InstrReg.RS2):
                m.d.comb += self.z_reg.eq(self._rs2)
            with m.Case(InstrReg.RD):
                m.d.comb += self.z_reg.eq(self._rd)

    def multiplex_to_reg_nums_chips(self, m: Module):
        mux_x = IC_mux32(4)
        mux_y = IC_mux32(4)
        mux_z = IC_mux32(4)
        m.submodules += [mux_x, mux_y, mux_z]
        m.d.comb += self.x_reg.eq(mux_x.y)
        m.d.comb += self.y_reg.eq(mux_y.y)
        m.d.comb += self.z_reg.eq(mux_z.y)

        sels = [InstrReg.ZERO, InstrReg.RS1, InstrReg.RS2, InstrReg.RD]
        sigs = [0, self._rs1, self._rs2, self._rd]

        for i in range(4):
            m.d.comb += mux_x.n_sel[i].eq(self._x_reg_select != sels[i])
            m.d.comb += mux_x.a[i].eq(sigs[i])
            m.d.comb += mux_y.n_sel[i].eq(self._y_reg_select != sels[i])
            m.d.comb += mux_y.a[i].eq(sigs[i])
            m.d.comb += mux_z.n_sel[i].eq(self._z_reg_select != sels[i])
            m.d.comb += mux_z.a[i].eq(sigs[i])

    def decode_imm(self, m: Module):
        """Decodes the immediate value out of the instruction."""
        with m.Switch(self._imm_format):
            # Format I instructions. Surprisingly, SLTIU (Set if Less Than
            # Immediate Unsigned) actually does sign-extend the immediate
            # value, and then compare as if the sign-extended immediate value
            # were unsigned!
            with m.Case(OpcodeFormat.I):
                tmp = Signal(signed(12))
                m.d.comb += tmp.eq(self.state._instr[20:])
                m.d.comb += self._imm.eq(tmp)

            # Format S instructions:
            with m.Case(OpcodeFormat.S):
                tmp = Signal(signed(12))
                m.d.comb += tmp[0:5].eq(self.state._instr[7:12])
                m.d.comb += tmp[5:].eq(self.state._instr[25:])
                m.d.comb += self._imm.eq(tmp)

            # Format R instructions:
            with m.Case(OpcodeFormat.R):
                m.d.comb += self._imm.eq(0)

            # Format U instructions:
            with m.Case(OpcodeFormat.U):
                m.d.comb += self._imm.eq(0)
                m.d.comb += self._imm[12:].eq(self.state._instr[12:])

            # Format B instructions:
            with m.Case(OpcodeFormat.B):
                tmp = Signal(signed(13))
                m.d.comb += [
                    tmp[12].eq(self.state._instr[31]),
                    tmp[11].eq(self.state._instr[7]),
                    tmp[5:11].eq(self.state._instr[25:31]),
                    tmp[1:5].eq(self.state._instr[8:12]),
                    tmp[0].eq(0),
                    self._imm.eq(tmp),
                ]

            # Format J instructions:
            with m.Case(OpcodeFormat.J):
                tmp = Signal(signed(21))
                m.d.comb += [
                    tmp[20].eq(self.state._instr[31]),
                    tmp[12:20].eq(self.state._instr[12:20]),
                    tmp[11].eq(self.state._instr[20]),
                    tmp[1:11].eq(self.state._instr[21:31]),
                    tmp[0].eq(0),
                    self._imm.eq(tmp),
                ]

            with m.Case(OpcodeFormat.SYS):
                m.d.comb += [
                    self._imm[0:5].eq(self.state._instr[15:]),
                    self._imm[5:].eq(0),
                ]

    def decode_imm_chips(self, m: Module):
        mux = IC_mux32(6)
        gal = IC_GAL_imm_format_decoder()

        m.submodules += mux
        m.submodules += gal

        m.d.comb += gal.opcode.eq(self._opcode)
        m.d.comb += mux.n_sel[0].eq(gal.i_n_oe)
        m.d.comb += mux.n_sel[1].eq(gal.s_n_oe)
        m.d.comb += mux.n_sel[2].eq(gal.u_n_oe)
        m.d.comb += mux.n_sel[3].eq(gal.b_n_oe)
        m.d.comb += mux.n_sel[4].eq(gal.j_n_oe)
        m.d.comb += mux.n_sel[5].eq(gal.sys_n_oe)

        instr = self.state._instr

        # Format I
        m.d.comb += [
            mux.a[0][0:12].eq(instr[20:]),
            mux.a[0][12:].eq(Repl(instr[31], 32)),  # sext
        ]

        # Format S
        m.d.comb += [
            mux.a[1][0:5].eq(instr[7:]),
            mux.a[1][5:11].eq(instr[25:]),
            mux.a[1][11:].eq(Repl(instr[31], 32)),  # sext
        ]

        # Format U
        m.d.comb += [
            mux.a[2][0:12].eq(0),
            mux.a[2][12:].eq(instr[12:]),
        ]

        # Format B
        m.d.comb += [
            mux.a[3][0].eq(0),
            mux.a[3][1:5].eq(instr[8:]),
            mux.a[3][5:11].eq(instr[25:]),
            mux.a[3][11].eq(instr[7]),
            mux.a[3][12:].eq(Repl(instr[31], 32)),  # sext
        ]

        # Format J
        m.d.comb += [
            mux.a[4][0].eq(0),
            mux.a[4][1:11].eq(instr[21:]),
            mux.a[4][11].eq(instr[20]),
            mux.a[4][12:20].eq(instr[12:]),
            mux.a[4][20:].eq(Repl(instr[31], 32)),  # sext
        ]

        # Format SYS
        m.d.comb += [
            mux.a[5][0:5].eq(instr[15:]),
            mux.a[5][5:].eq(0),
        ]

        m.d.comb += self._imm.eq(mux.y)

    def handle_trap(self, m: Module):
        """Adds trap handling logic.

        For fatals, we store the cause and then halt.

        TODO: This code is horrible because some of it doesn't use multiplexers.
        """
        is_int = ~self.state._exception

        fatal = Signal()
        m.d.comb += fatal.eq(all_true(~is_int,
                                      ~self.state._mcause[31],
                                      self.state._mcause != TrapCause.EXC_ECALL_FROM_MACH_MODE,
                                      self.state._mcause != TrapCause.EXC_BREAKPOINT))

        with m.If(is_int):
            with m.If(self.state._mip[MInterrupt.MEI]):
                m.d.comb += self._trapcause.eq(TrapCause.INT_MACH_EXTERNAL)
                m.d.comb += self._trapcause_to_mcause.eq(1)
                m.d.comb += self._clear_pend_mei.eq(1)
            with m.Elif(self.state._mip[MInterrupt.MTI]):
                m.d.comb += self._trapcause.eq(TrapCause.INT_MACH_TIMER)
                m.d.comb += self._trapcause_to_mcause.eq(1)
                m.d.comb += self._clear_pend_mti.eq(1)

            m.d.comb += self._pc_to_mepc.eq(1)

        with m.If(fatal):
            m.d.comb += self._next_instr_phase.eq(0)
            m.d.comb += self.fatal.eq(1)

        vec_mode = self.state._mtvec[:2]

        m.d.comb += self.data_x_out.eq(self.state._mtvec & 0xFFFFFFFC)
        with m.If(all_true(is_int, vec_mode == 1)):
            m.d.comb += [
                self.data_y_out[2:].eq(self.state._mcause[:30]),
                self.data_y_out[0:2].eq(0),
            ]
        with m.Else():
            m.d.comb += self.data_y_out.eq(0)

        m.d.comb += [
            self.alu_op_to_z.eq(AluOp.ADD),
            self._z_to_memaddr.eq(1),
            self._z_to_pc.eq(1),
            self.instr_complete.eq(self.mcycle_end),
        ]

        with m.If(~fatal):
            m.d.ph1 += self.state.trap.eq(0)
            m.d.comb += self._enter_trap.eq(1)

    def handle_lui(self, m: Module):
        """Adds the LUI logic to the given module.

        rd <- r0 + imm
        PC <- PC + 4

        r0      -> X
        imm     -> Y
        ALU ADD -> Z
        Z       -> rd
        PC + 4  -> PC
        PC + 4  -> memaddr
        """
        m.d.comb += [
            self._imm_format.eq(OpcodeFormat.U),
            self.reg_to_x.eq(1),
            self._x_reg_select.eq(InstrReg.ZERO),
            self._imm_to_y.eq(1),
            self.alu_op_to_z.eq(AluOp.ADD),
            self._z_reg_select.eq(InstrReg.RD),
        ]
        m.d.comb += self._is_last_instr_cycle.eq(1)

    def handle_auipc(self, m: Module):
        """Adds the AUIPC logic to the given module.

        rd <- PC + imm
        PC <- PC + 4

        PC      -> X
        imm     -> Y
        ALU ADD -> Z
        Z       -> rd
        PC + 4  -> PC
        PC + 4  -> memaddr
        """
        m.d.comb += [
            self._imm_format.eq(OpcodeFormat.U),
            self._pc_to_x.eq(1),
            self._imm_to_y.eq(1),
            self.alu_op_to_z.eq(AluOp.ADD),
            self._z_reg_select.eq(InstrReg.RD),
        ]
        m.d.comb += self._is_last_instr_cycle.eq(1)

    def handle_op_imm(self, m: Module):
        """Adds the OP_IMM logic to the given module.

        rd <- rs1 op imm
        PC <- PC + 4

        rs1     -> X
        imm     -> Y
        ALU op  -> Z
        Z       -> rd
        PC + 4  -> PC
        PC + 4  -> memaddr
        """
        m.d.comb += [
            self._imm_format.eq(OpcodeFormat.I),
            self.reg_to_x.eq(1),
            self._x_reg_select.eq(InstrReg.RS1),
            self._imm_to_y.eq(1),
            self._z_reg_select.eq(InstrReg.RD),
        ]
        with m.Switch(self._alu_func):
            with m.Case(AluFunc.ADD):
                m.d.comb += self.alu_op_to_z.eq(AluOp.ADD)
            with m.Case(AluFunc.SUB):
                m.d.comb += self.alu_op_to_z.eq(AluOp.SUB)
            with m.Case(AluFunc.SLL):
                m.d.comb += self.alu_op_to_z.eq(AluOp.SLL)
            with m.Case(AluFunc.SLT):
                m.d.comb += self.alu_op_to_z.eq(AluOp.SLT)
            with m.Case(AluFunc.SLTU):
                m.d.comb += self.alu_op_to_z.eq(AluOp.SLTU)
            with m.Case(AluFunc.XOR):
                m.d.comb += self.alu_op_to_z.eq(AluOp.XOR)
            with m.Case(AluFunc.SRL):
                m.d.comb += self.alu_op_to_z.eq(AluOp.SRL)
            with m.Case(AluFunc.SRA):
                m.d.comb += self.alu_op_to_z.eq(AluOp.SRA)
            with m.Case(AluFunc.OR):
                m.d.comb += self.alu_op_to_z.eq(AluOp.OR)
            with m.Case(AluFunc.AND):
                m.d.comb += self.alu_op_to_z.eq(AluOp.AND)
            with m.Default():
                self.handle_illegal_instr(m)
        m.d.comb += self._is_last_instr_cycle.eq(1)

    def handle_op(self, m: Module):
        """Adds the OP logic to the given module.

        rd <- rs1 op rs2
        PC <- PC + 4

        rs1     -> X
        rs2     -> Y
        ALU op  -> Z
        Z       -> rd
        PC + 4  -> PC
        PC + 4  -> memaddr
        """
        m.d.comb += [
            self._imm_format.eq(OpcodeFormat.R),
            self.reg_to_x.eq(1),
            self._x_reg_select.eq(InstrReg.RS1),
            self.reg_to_y.eq(1),
            self._y_reg_select.eq(InstrReg.RS2),
            self._z_reg_select.eq(InstrReg.RD),
        ]
        with m.Switch(self._alu_func):
            with m.Case(AluFunc.ADD):
                m.d.comb += self.alu_op_to_z.eq(AluOp.ADD)
            with m.Case(AluFunc.SUB):
                m.d.comb += self.alu_op_to_z.eq(AluOp.SUB)
            with m.Case(AluFunc.SLL):
                m.d.comb += self.alu_op_to_z.eq(AluOp.SLL)
            with m.Case(AluFunc.SLT):
                m.d.comb += self.alu_op_to_z.eq(AluOp.SLT)
            with m.Case(AluFunc.SLTU):
                m.d.comb += self.alu_op_to_z.eq(AluOp.SLTU)
            with m.Case(AluFunc.XOR):
                m.d.comb += self.alu_op_to_z.eq(AluOp.XOR)
            with m.Case(AluFunc.SRL):
                m.d.comb += self.alu_op_to_z.eq(AluOp.SRL)
            with m.Case(AluFunc.SRA):
                m.d.comb += self.alu_op_to_z.eq(AluOp.SRA)
            with m.Case(AluFunc.OR):
                m.d.comb += self.alu_op_to_z.eq(AluOp.OR)
            with m.Case(AluFunc.AND):
                m.d.comb += self.alu_op_to_z.eq(AluOp.AND)
            with m.Default():
                self.handle_illegal_instr(m)
        m.d.comb += self._is_last_instr_cycle.eq(1)

    def handle_jal(self, m: Module):
        """Adds the JAL logic to the given module.

        rd <- PC + 4, PC <- PC + imm

        PC      -> X
        imm     -> Y
        ALU ADD -> Z
        Z       -> memaddr
        ---------------------
        PC + 4  -> Z
        Z       -> rd
        memaddr -> PC   # This will zero the least significant bit

        Note that because the immediate value for JAL has its least
        significant bit set to zero by definition, and the PC is also
        assumed to be aligned, there is no loss in generality to clear
        the least significant bit when transferring memaddr to PC.
        """
        m.d.comb += self._imm_format.eq(OpcodeFormat.J)

        with m.If(self.state._instr_phase == 0):
            m.d.comb += [
                self._pc_to_x.eq(1),
                self._imm_to_y.eq(1),
                self.alu_op_to_z.eq(AluOp.ADD),
                self._z_to_memaddr.eq(1),
                self._next_instr_phase.eq(1),
            ]
        with m.Else():
            with m.If(self.state.memaddr[1] != 0):
                self.set_exception(
                    m, TrapCause.EXC_INSTR_ADDR_MISALIGN, mtval=self._memaddr_to_mtval)
            with m.Else():
                m.d.comb += [
                    self._pc_plus_4_to_z.eq(1),
                    self._z_reg_select.eq(InstrReg.RD),
                    self._memaddr_to_pc.eq(1),
                    self._is_last_instr_cycle.eq(1),
                ]

    def handle_jalr(self, m: Module):
        """Adds the JALR logic to the given module.

        rd <- PC + 4, PC <- (rs1 + imm) & 0xFFFFFFFE

        rs1     -> X
        imm     -> Y
        ALU ADD -> Z
        Z       -> memaddr
        ---------------------
        PC + 4  -> Z
        Z       -> rd
        memaddr -> PC  # This will zero the least significant bit
        """
        m.d.comb += self._imm_format.eq(OpcodeFormat.J)

        with m.If(self.state._instr_phase == 0):
            m.d.comb += [
                self.reg_to_x.eq(1),
                self._x_reg_select.eq(InstrReg.RS1),
                self._imm_to_y.eq(1),
                self.alu_op_to_z.eq(AluOp.ADD),
                self._z_to_memaddr.eq(1),
                self._next_instr_phase.eq(1),
            ]
        with m.Else():
            with m.If(self.state.memaddr[1] != 0):
                self.set_exception(
                    m, TrapCause.EXC_INSTR_ADDR_MISALIGN, mtval=self._memaddr_lsb_masked_to_mtval)
            with m.Else():
                m.d.comb += [
                    self._pc_plus_4_to_z.eq(1),
                    self._z_reg_select.eq(InstrReg.RD),
                    self._memaddr_to_pc.eq(1),
                    self._is_last_instr_cycle.eq(1),
                ]

    def handle_branch(self, m: Module):
        """Adds the BRANCH logic to the given module.

        cond <- rs1 - rs2 < 0, rs1 - rs2 == 0
        if f(cond):
            PC <- PC + imm
        else:
            PC <- PC + 4

        rs1     -> X
        rs2     -> Y
        ALU SUB -> Z, cond
        --------------------- cond == 1
        PC      -> X
        imm/4   -> Y (imm for cond == 1, 4 otherwise)
        ALU ADD -> Z
        Z       -> PC
        Z       -> memaddr
        --------------------- cond == 0
        PC + 4  -> PC
        PC + 4  -> memaddr
        """
        m.d.comb += self._imm_format.eq(OpcodeFormat.B)

        with m.If(self.state._instr_phase == 0):
            m.d.comb += [
                self.reg_to_x.eq(1),
                self._x_reg_select.eq(InstrReg.RS1),
                self.reg_to_y.eq(1),
                self._y_reg_select.eq(InstrReg.RS2),
                self.alu_op_to_z.eq(AluOp.SUB),
                self._next_instr_phase.eq(1),
            ]
        with m.Else():
            cond = Signal()
            with m.Switch(self._funct3):
                with m.Case(BranchCond.EQ):
                    m.d.comb += cond.eq(self.state._stored_alu_eq == 1)
                with m.Case(BranchCond.NE):
                    m.d.comb += cond.eq(self.state._stored_alu_eq == 0)
                with m.Case(BranchCond.LT):
                    m.d.comb += cond.eq(self.state._stored_alu_lt == 1)
                with m.Case(BranchCond.GE):
                    m.d.comb += cond.eq(self.state._stored_alu_lt == 0)
                with m.Case(BranchCond.LTU):
                    m.d.comb += cond.eq(self.state._stored_alu_ltu == 1)
                with m.Case(BranchCond.GEU):
                    m.d.comb += cond.eq(self.state._stored_alu_ltu == 0)
                with m.Default():
                    self.handle_illegal_instr(m)

            with m.If(cond):
                m.d.comb += self._imm_to_y.eq(1)
            with m.Else():
                m.d.comb += self._shamt.eq(4)
                m.d.comb += self._shamt_to_y.eq(1)

            m.d.comb += [
                self._pc_to_x.eq(1),
                self.alu_op_to_z.eq(AluOp.ADD),
            ]

            with m.If(self.data_z_in[0:2] != 0):
                self.set_exception(
                    m, TrapCause.EXC_INSTR_ADDR_MISALIGN, mtval=self._z_to_mtval)
            with m.Else():
                m.d.comb += [
                    self._z_to_pc.eq(1),
                    self._z_to_memaddr.eq(1),
                    self._is_last_instr_cycle.eq(1),
                ]

    def handle_load(self, m: Module):
        """Adds the LOAD logic to the given module.

        Note that byte loads are byte-aligned, half-word loads
        are 16-bit aligned, and word loads are 32-bit aligned.
        Attempting to load unaligned will lead to undefined
        behavior.

        Operation is to load 32 bits from a 32-bit aligned
        address, and then perform at most two shifts to get
        the desired behavior: a shift left to get the most
        significant byte into the leftmost position, then a
        shift right to zero or sign extend the value.

        For example, for loading a half-word starting at
        address A where A%4=0, we first load the full 32
        bits at that address, resulting in XYHL, where X and
        Y are unwanted and H and L are the half-word we want
        to load. Then we shift left by 16: HL00. And finally
        we shift right by 16, either signed or unsigned
        depending on whether we are doing an LH or an LHU:
        ssHL / 00HL.

        addr <- rs1 + imm
        rd <- data at addr, possibly sign-extended
        PC <- PC + 4

        If we let N be addr%4, then:

        instr   N   shift1  shift2
        --------------------------
        LB      0   SLL 24  SRA 24
        LB      1   SLL 16  SRA 24
        LB      2   SLL  8  SRA 24
        LB      3   SLL  0  SRA 24
        LBU     0   SLL 24  SRL 24
        LBU     1   SLL 16  SRL 24
        LBU     2   SLL  8  SRL 24
        LBU     3   SLL  0  SRL 24
        LH      0   SLL 16  SRA 16
        LH      2   SLL  0  SRA 16
        LHU     0   SLL 16  SRL 16
        LHU     2   SLL  0  SRL 16
        LW      0   SLL  0  SRA  0
        (all other N are misaligned accesses)

        Where there is an SLL 0, the machine cycle
        could be skipped, but in the interests of
        simpler logic, we will not do that.

        rs1     -> X
        imm     -> Y
        ALU ADD -> Z
        Z       -> memaddr
        ---------------------
        memdata -> X
        shamt1  -> Y
        ALU SLL -> Z
        Z       -> rd
        ---------------------
        rd          -> X
        shamt2      -> Y
        ALU SRA/SRL -> Z
        Z           -> rd
        PC + 4      -> PC
        PC + 4      -> memaddr
        """
        m.d.comb += self._imm_format.eq(OpcodeFormat.I)

        with m.If(self.state._instr_phase == 0):
            m.d.comb += [
                self.reg_to_x.eq(1),
                self._x_reg_select.eq(InstrReg.RS1),
                self._imm_to_y.eq(1),
                self.alu_op_to_z.eq(AluOp.ADD),
                self._z_to_memaddr.eq(1),
                self._next_instr_phase.eq(1),
            ]

        with m.Elif(self.state._instr_phase == 1):
            m.d.comb += [
                self.mem_rd.eq(1),
                self._memdata_to_x.eq(1),
                self._shamt_to_y.eq(1),
                self.alu_op_to_z.eq(AluOp.SLL),
                self._z_reg_select.eq(InstrReg.RD),
                self._next_instr_phase.eq(2),
            ]

            with m.Switch(self._funct3):

                with m.Case(MemAccessWidth.B, MemAccessWidth.BU):
                    with m.Switch(self.state.memaddr[0:2]):
                        with m.Case(0):
                            m.d.comb += self._shamt.eq(24)
                        with m.Case(1):
                            m.d.comb += self._shamt.eq(16)
                        with m.Case(2):
                            m.d.comb += self._shamt.eq(8)
                        with m.Case(3):
                            m.d.comb += self._shamt.eq(0)

                with m.Case(MemAccessWidth.H, MemAccessWidth.HU):
                    with m.Switch(self.state.memaddr[0:2]):
                        with m.Case(0):
                            m.d.comb += self._shamt.eq(16)
                        with m.Case(2):
                            m.d.comb += self._shamt.eq(0)
                        with m.Default():
                            self.set_exception(
                                m, TrapCause.EXC_LOAD_ADDR_MISALIGN, mtval=self._memaddr_to_mtval)

                with m.Case(MemAccessWidth.W):
                    with m.Switch(self.state.memaddr[0:2]):
                        with m.Case(0):
                            m.d.comb += self._shamt.eq(0)
                        with m.Default():
                            self.set_exception(
                                m, TrapCause.EXC_LOAD_ADDR_MISALIGN, mtval=self._memaddr_to_mtval)

                with m.Default():
                    self.handle_illegal_instr(m)

        with m.Else():
            m.d.comb += [
                self.reg_to_x.eq(1),
                self._x_reg_select.eq(InstrReg.RD),
                self._shamt_to_y.eq(1),
                self._z_reg_select.eq(InstrReg.RD),
            ]

            with m.Switch(self._funct3):
                with m.Case(MemAccessWidth.B):
                    m.d.comb += [
                        self._shamt.eq(24),
                        self.alu_op_to_z.eq(AluOp.SRA),
                    ]
                with m.Case(MemAccessWidth.BU):
                    m.d.comb += [
                        self._shamt.eq(24),
                        self.alu_op_to_z.eq(AluOp.SRL),
                    ]
                with m.Case(MemAccessWidth.H):
                    m.d.comb += [
                        self._shamt.eq(16),
                        self.alu_op_to_z.eq(AluOp.SRA),
                    ]
                with m.Case(MemAccessWidth.HU):
                    m.d.comb += [
                        self._shamt.eq(16),
                        self.alu_op_to_z.eq(AluOp.SRL),
                    ]
                with m.Case(MemAccessWidth.W):
                    m.d.comb += [
                        self._shamt.eq(0),
                        self.alu_op_to_z.eq(AluOp.SRL),
                    ]

            m.d.comb += self._is_last_instr_cycle.eq(1)

    def handle_store(self, m: Module):
        """Adds the STORE logic to the given module.

        Note that byte stores are byte-aligned, half-word stores
        are 16-bit aligned, and word stores are 32-bit aligned.
        Attempting to stores unaligned will lead to undefined
        behavior.

        addr <- rs1 + imm
        data <- rs2
        PC <- PC + 4

        rs1     -> X
        imm     -> Y
        ALU ADD -> Z
        Z       -> memaddr
        ---------------------
        rs2     -> X
        shamt   -> Y
        ALU SLL -> Z
        Z       -> wrdata
                -> wrmask
        ---------------------
        PC + 4  -> PC
        PC + 4  -> memaddr
        """
        m.d.comb += self._imm_format.eq(OpcodeFormat.S)

        with m.If(self.state._instr_phase == 0):
            m.d.comb += [
                self.reg_to_x.eq(1),
                self._x_reg_select.eq(InstrReg.RS1),
                self._imm_to_y.eq(1),
                self.alu_op_to_z.eq(AluOp.ADD),
                self._z_to_memaddr.eq(1),
                self._next_instr_phase.eq(1),
            ]

        with m.Elif(self.state._instr_phase == 1):
            m.d.comb += [
                self.reg_to_x.eq(1),
                self._x_reg_select.eq(InstrReg.RS2),
                self._shamt_to_y.eq(1),
                self.alu_op_to_z.eq(AluOp.SLL),
                self._z_to_memdata.eq(1),
                self._next_instr_phase.eq(2),
            ]

            with m.Switch(self._funct3):

                with m.Case(MemAccessWidth.B):
                    with m.Switch(self.state.memaddr[0:2]):
                        with m.Case(0):
                            m.d.comb += self._shamt.eq(0)
                        with m.Case(1):
                            m.d.comb += self._shamt.eq(8)
                        with m.Case(2):
                            m.d.comb += self._shamt.eq(16)
                        with m.Case(3):
                            m.d.comb += self._shamt.eq(24)

                with m.Case(MemAccessWidth.H):
                    with m.Switch(self.state.memaddr[0:2]):
                        with m.Case(0):
                            m.d.comb += self._shamt.eq(0)
                        with m.Case(2):
                            m.d.comb += self._shamt.eq(16)
                        with m.Default():
                            self.set_exception(
                                m, TrapCause.EXC_STORE_AMO_ADDR_MISALIGN, mtval=self._memaddr_to_mtval)

                with m.Case(MemAccessWidth.W):
                    with m.Switch(self.state.memaddr[0:2]):
                        with m.Case(0):
                            m.d.comb += self._shamt.eq(0)
                        with m.Default():
                            self.set_exception(
                                m, TrapCause.EXC_STORE_AMO_ADDR_MISALIGN, mtval=self._memaddr_to_mtval)

                with m.Default():
                    self.handle_illegal_instr(m)

        with m.Else():
            with m.Switch(self._funct3):

                with m.Case(MemAccessWidth.B):
                    with m.Switch(self.state.memaddr[0:2]):
                        with m.Case(0):
                            m.d.comb += self.mem_wr_mask.eq(0b0001)
                        with m.Case(1):
                            m.d.comb += self.mem_wr_mask.eq(0b0010)
                        with m.Case(2):
                            m.d.comb += self.mem_wr_mask.eq(0b0100)
                        with m.Case(3):
                            m.d.comb += self.mem_wr_mask.eq(0b1000)

                with m.Case(MemAccessWidth.H):
                    with m.Switch(self.state.memaddr[0:2]):
                        with m.Case(0):
                            m.d.comb += self.mem_wr_mask.eq(0b0011)
                        with m.Case(2):
                            m.d.comb += self.mem_wr_mask.eq(0b1100)

                with m.Case(MemAccessWidth.W):
                    with m.Switch(self.state.memaddr[0:2]):
                        with m.Case(0):
                            m.d.comb += self.mem_wr_mask.eq(0b1111)
            m.d.comb += self.mem_wr.eq(1)
            m.d.comb += self._is_last_instr_cycle.eq(1)

    def handle_system(self, m: Module):
        """Adds the SYSTEM logic to the given module.

        Some points of interest:

        * Attempts to write a read-only register
          result in an illegal instruction exception.
        * Attempts to access a CSR that doesn't exist
          result in an illegal instruction exception.
        * Attempts to write read-only bits to a read/write CSR
          are ignored.

        Because we're building this in hardware, which is
        expensive, we're not implementing any CSRs that aren't
        strictly necessary. The documentation for the misa, mvendorid,
        marchid, and mimpid registers state that they can return zero if
        unimplemented. This implies that unimplemented CSRs still
        exist.

        The mhartid, because we only have one HART, can just return zero.
        """
        m.d.comb += self._imm_format.eq(OpcodeFormat.SYS)

        with m.Switch(self._funct3):
            with m.Case(SystemFunc.CSRRW):
                self.handle_CSRRW(m)
            with m.Case(SystemFunc.CSRRWI):
                self.handle_CSRRWI(m)
            with m.Case(SystemFunc.CSRRS):
                self.handle_CSRRS(m)
            with m.Case(SystemFunc.CSRRSI):
                self.handle_CSRRSI(m)
            with m.Case(SystemFunc.CSRRC):
                self.handle_CSRRC(m)
            with m.Case(SystemFunc.CSRRCI):
                self.handle_CSRRCI(m)
            with m.Case(SystemFunc.PRIV):
                self.handle_PRIV(m)
            with m.Default():
                self.handle_illegal_instr(m)

    def handle_CSRRW(self, m: Module):
        m.d.comb += self._funct12_to_csr_num.eq(1)

        with m.If(self.state._instr_phase == 0):
            with m.If(self._rd == 0):
                m.d.comb += [
                    self._x_reg_select.eq(InstrReg.ZERO),
                    self.reg_to_x.eq(1),
                ]
            with m.Else():
                m.d.comb += [
                    self.csr_to_x.eq(1)
                ]
            m.d.comb += [
                self._y_reg_select.eq(InstrReg.RS1),
                self.reg_to_y.eq(1),
                self.alu_op_to_z.eq(AluOp.Y),
                self.z_to_csr.eq(1),
                self._x_to_tmp.eq(1),
                self._next_instr_phase.eq(1),
            ]

        with m.Else():
            m.d.comb += [
                self._tmp_to_z.eq(1),
                self._z_reg_select.eq(InstrReg.RD),
            ]
            m.d.comb += self._is_last_instr_cycle.eq(1)

    def handle_CSRRWI(self, m: Module):
        m.d.comb += self._funct12_to_csr_num.eq(1)

        with m.If(self.state._instr_phase == 0):
            with m.If(self._rd == 0):
                m.d.comb += [
                    self._x_reg_select.eq(InstrReg.ZERO),
                    self.reg_to_x.eq(1),
                ]
            with m.Else():
                m.d.comb += [
                    self.csr_to_x.eq(1)
                ]
            m.d.comb += [
                self._imm_to_y.eq(1),
                self.alu_op_to_z.eq(AluOp.Y),
                self.z_to_csr.eq(1),
                self._x_to_tmp.eq(1),
                self._next_instr_phase.eq(1),
            ]

        with m.Else():
            m.d.comb += [
                self._tmp_to_z.eq(1),
                self._z_reg_select.eq(InstrReg.RD),
            ]
            m.d.comb += self._is_last_instr_cycle.eq(1)

    def handle_CSRRS(self, m: Module):
        m.d.comb += self._funct12_to_csr_num.eq(1)

        with m.If(self.state._instr_phase == 0):
            m.d.comb += [
                self.csr_to_x.eq(1),
                self._y_reg_select.eq(InstrReg.RS1),
                self.reg_to_y.eq(1),
                self.alu_op_to_z.eq(AluOp.OR),
                self.z_to_csr.eq(self._rs1 != 0),
                self._x_to_tmp.eq(1),
                self._next_instr_phase.eq(1),
            ]

        with m.Else():
            m.d.comb += [
                self._tmp_to_z.eq(1),
                self._z_reg_select.eq(InstrReg.RD),
            ]
            m.d.comb += self._is_last_instr_cycle.eq(1)

    def handle_CSRRSI(self, m: Module):
        m.d.comb += self._funct12_to_csr_num.eq(1)

        with m.If(self.state._instr_phase == 0):
            m.d.comb += [
                self.csr_to_x.eq(1),
                self._imm_to_y.eq(1),
                self.alu_op_to_z.eq(AluOp.OR),
                self.z_to_csr.eq(self._imm != 0),
                self._x_to_tmp.eq(1),
                self._next_instr_phase.eq(1),
            ]

        with m.Else():
            m.d.comb += [
                self._tmp_to_z.eq(1),
                self._z_reg_select.eq(InstrReg.RD),
            ]
            m.d.comb += self._is_last_instr_cycle.eq(1)

    def handle_CSRRC(self, m: Module):
        m.d.comb += self._funct12_to_csr_num.eq(1)

        with m.If(self.state._instr_phase == 0):
            m.d.comb += [
                self.csr_to_x.eq(1),
                self._y_reg_select.eq(InstrReg.RS1),
                self.reg_to_y.eq(1),
                self.alu_op_to_z.eq(AluOp.AND_NOT),
                self.z_to_csr.eq(self._rs1 != 0),
                self._x_to_tmp.eq(1),
                self._next_instr_phase.eq(1),
            ]

        with m.Else():
            m.d.comb += [
                self._tmp_to_z.eq(1),
                self._z_reg_select.eq(InstrReg.RD),
            ]
            m.d.comb += self._is_last_instr_cycle.eq(1)

    def handle_CSRRCI(self, m: Module):
        m.d.comb += self._funct12_to_csr_num.eq(1)

        with m.If(self.state._instr_phase == 0):
            m.d.comb += [
                self.csr_to_x.eq(1),
                self._imm_to_y.eq(1),
                self.alu_op_to_z.eq(AluOp.AND_NOT),
                self.z_to_csr.eq(self._imm != 0),
                self._x_to_tmp.eq(1),
                self._next_instr_phase.eq(1),
            ]

        with m.Else():
            m.d.comb += [
                self._tmp_to_z.eq(1),
                self._z_reg_select.eq(InstrReg.RD),
            ]
            m.d.comb += self._is_last_instr_cycle.eq(1)

    def handle_PRIV(self, m: Module):
        with m.Switch(self._funct12):
            with m.Case(PrivFunc.MRET):
                with m.If((self._rd == 0) & (self._rs1 == 0)):
                    self.handle_MRET(m)
                with m.Else():
                    self.handle_illegal_instr(m)
            with m.Case(PrivFunc.ECALL):
                with m.If((self._rd == 0) & (self._rs1 == 0)):
                    self.handle_ECALL(m)
                with m.Else():
                    self.handle_illegal_instr(m)
            with m.Case(PrivFunc.EBREAK):
                with m.If((self._rd == 0) & (self._rs1 == 0)):
                    self.handle_EBREAK(m)
                with m.Else():
                    self.handle_illegal_instr(m)
            with m.Default():
                self.handle_illegal_instr(m)

    def handle_MRET(self, m: Module):
        m.d.comb += [
            # TODO: Change to _mepc_to_pc
            self._mepc_num_to_csr_num.eq(1),
            self.csr_to_x.eq(1),
            self._x_to_pc.eq(1),
            self._exit_trap.eq(1),
            self._is_last_instr_cycle.eq(1),
        ]

    def handle_ECALL(self, m: Module):
        """Handles the ECALL instruction.

        Note that normally, ECALL is used from a lower privelege mode, which stores
        the PC of the instruction in the appropriate lower EPC CSR (e.g. SEPC or UEPC).
        This allows interrupts to be handled during the call, because we're in a higher
        privelege level. However, in machine mode, there is no higher privelege level,
        so we have no choice but to disable interrupts for an ECALL.
        """
        self.set_exception(
            m, TrapCause.EXC_ECALL_FROM_MACH_MODE, mtval=self._pc_to_mtval, fatal=False)

    def handle_EBREAK(self, m: Module):
        """Handles the EBREAK instruction.

        Note that normally, EBREAK is used from a lower privelege mode, which stores
        the PC of the instruction in the appropriate lower EPC CSR (e.g. SEPC or UEPC).
        This allows interrupts to be handled during the call, because we're in a higher
        privelege level. However, in machine mode, there is no higher privelege level,
        so we have no choice but to disable interrupts for an EBREAK.
        """
        self.set_exception(
            m, TrapCause.EXC_BREAKPOINT, mtval=self._pc_to_mtval, fatal=False)

    @ classmethod
    def formal(cls) -> Tuple[Module, List[Signal]]:
        """Formal verification for the sequencer."""
        m = Module()

        ph1 = ClockDomain("ph1")
        ph2 = ClockDomain("ph2")
        seq = SequencerCard()

        m.domains += [ph1, ph2]
        m.submodules += seq

        # Generate the ph1 and ph2 clocks.
        cycle_count = Signal(8, reset_less=True)
        phase_count = Signal(3, reset_less=True)

        m.d.sync += phase_count.eq(phase_count + 1)
        with m.If(phase_count == 5):
            m.d.sync += phase_count.eq(0)
            m.d.sync += cycle_count.eq(cycle_count + 1)

        with m.Switch(phase_count):
            with m.Case(0, 1, 2):
                m.d.comb += ph1.clk.eq(1)
            with m.Default():
                m.d.comb += ph1.clk.eq(0)

        with m.Switch(phase_count):
            with m.Case(1, 4):
                m.d.comb += ph2.clk.eq(0)
            with m.Default():
                m.d.comb += ph2.clk.eq(1)

        m.d.comb += seq.mcycle_end.eq(phase_count == 5)

        m.d.comb += Cover(all_true(seq.instr_complete,
                                   seq._opcode == Opcode.STORE))
        m.d.comb += Cover(seq.state._pc > 0x100)
        m.d.comb += Cover(all_true(cycle_count > 1,
                                   Past(seq.fatal, 6), Past(seq.fatal, 12), seq.fatal))

        with m.If(phase_count > 1):
            m.d.comb += [
                Assume(Stable(seq.memdata_rd)),
                Assume(Stable(seq.data_x_in)),
                Assume(Stable(seq.data_y_in)),
                Assume(Stable(seq.data_z_in)),
                Assume(Stable(seq.alu_eq)),
                Assume(Stable(seq.alu_lt)),
                Assume(Stable(seq.alu_ltu)),
            ]

            m.d.comb += [
                Assert(Stable(seq.x_reg)),
                Assert(Stable(seq.y_reg)),
                Assert(Stable(seq.z_reg)),
                Assert(Stable(seq.reg_to_x)),
                Assert(Stable(seq.reg_to_y)),
                Assert(Stable(seq.alu_op_to_z)),

                Assert(Stable(seq.data_x_out)),
                Assert(Stable(seq.data_y_out)),
                Assert(Stable(seq.data_z_out)),

                Assert(Stable(seq.mem_rd)),
                Assert(Stable(seq.mem_wr)),
                Assert(Stable(seq.mem_wr_mask)),
                Assert(Stable(seq.state.memaddr)),
                Assert(Stable(seq.state.memdata_wr)),
            ]

        return m, [seq.memdata_rd, seq.data_x_in, seq.data_y_in, seq.data_z_in] + [seq._pc_plus_4_to_memaddr, seq._x_to_memaddr, seq._z_to_memdata, seq._imm, seq._imm_format, seq._is_last_instr_cycle, seq.instr_complete]


if __name__ == "__main__":
    main(SequencerCard)
