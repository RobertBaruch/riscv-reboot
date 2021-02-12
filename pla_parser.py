# Minimally parses a PLA file.
import json
import os
import pprint
import sys

import pyeda

from typing import List, Dict

from munkres import Munkres, make_cost_matrix, DISALLOWED, print_matrix
from pyeda.inter import *

# PLA files that I'm interested in look like this:
#
# # Benchmark "top" written by ABC on Mon Jan 25 13:56:23 2021
# .i 12
# .o 8
# .ilb b[0] b[1] b[2] b[3] s[0] s[1] s[2] s[3] a[0] a[1] a[2] a[3]
# .ob x[0] x[1] x[2] x[3] x[4] x[5] x[6] x[7]
# .p 20
# 1---0---0--- 10000000
# 0----0--0--- 10000000
# ...more lines
# .e
#
# Some format information here:
#   http://www.ecs.umass.edu/ece/labs/vlsicad/ece667/links/espresso.5.html
#
# So the format spec is:
# A '#' in the first character of the line is a comment.
# .i %d:
#     Number of input variables
# .o %d:
#     Number of output functions
# .ilb <space-separated signal name list>:
#     Names of input variables. Must come after .i. There must be the same
#     number of names as there is in .i.
# .ob <space-separated signal name list>:
#     Names of output functions. Must come after .o. There must be the same
#     number of names as there is in .o.
# .p %d:
#     Number of product terms. May be ignored.
# .e or .end:
#     Optionally marks end of description.
# Product term line:
#     .i number of 1/0/- characters, followed by whitespace, followed by
#     .o number of 1/0 characters. These are in the same order as the
#     input and output names.

# Note that this kind of PLA file only represents and-or (aka sum-of-products).
# Because we're also interested in xor layers, and also multiple layers, we
# have to use multiples of these files, and also a custom file for xor layers.


class ProductTerm():
    ones: List[str]
    zeros: List[str]

    def __init__(self):
        # List of symbolic inputs
        self.ones = []
        self.zeros = []
        self.expr = expr(1)

    def __repr__(self):
        pp = pprint.PrettyPrinter(indent=4)
        return pp.pformat({'ones': self.ones, 'zeros': self.zeros})


class OrTerm():
    products: List[ProductTerm]

    def __init__(self):
        self.products = []
        self.expr = expr(0)

    def __repr__(self):
        pp = pprint.PrettyPrinter(indent=4)
        return pp.pformat({'or_products': self.products, 'expr': self.expr})


def get_database():
    for path in sys.path:
        file = os.path.join(path, "database.json")
        if os.path.isfile(file):
            with open(file) as f:
                return json.load(f)
    return None


class PLAParser():
    inputs: List[str]
    outputs: List[str]
    or_terms: Dict[str, OrTerm]
    # If the file is marked with .xor, it's an XOR layer.
    is_xor: bool
    # If the file is marked with .outputs, all outputs are routed to pins.
    is_outputs: bool

    def __init__(self, file: str):
        self.inputs = []
        self.outputs = []
        # A map of symbolic output to OrTerm (or Xor)
        self.or_terms = {}
        self.is_xor = False
        self.is_outputs = False

        with open(file) as f:
            for line in f.readlines():
                if not self.readline(line):
                    break
        print(f"Inputs  : {self.inputs}")
        print(f"Outputs : {self.outputs}")
        pp = pprint.PrettyPrinter(indent=4, depth=3)
        print(f"OR Terms:")
        pprint.pprint(self.or_terms)

    def readline(self, line: str) -> bool:
        """Returns if there are more lines to parse."""
        if len(line) == 0:
            return True
        if line.startswith('#'):
            return True
        if line.startswith(".i "):
            return True
        if line.startswith(".o "):
            return True
        if line.startswith(".p "):
            return True
        if line.startswith(".e") | line.startswith(".end"):
            return False
        if line.startswith(".xor"):
            assert not self.is_outputs
            self.is_xor = True
        if line.startswith(".outputs"):
            assert not self.is_xor
            self.is_outputs = True
        if line.startswith(".ilb "):
            self.inputs = line.split()[1:]
            return True
        if line.startswith(".ob "):
            self.outputs = line.split()[1:]
            for output in self.outputs:
                self.or_terms[output] = OrTerm()
            return True
        if line.startswith("1") | line.startswith("0") | line.startswith("-"):
            assert not self.is_outputs
            if self.is_xor:
                self.read_xor_term(line)
            else:
                self.read_or_term(line)
            return True
        return True

    def read_or_term(self, line: str):
        parts = line.split()
        assert len(parts) == 2
        assert len(parts[0]) == len(self.inputs)
        assert len(parts[1]) == len(self.outputs)

        inputs = parts[0]
        outputs = parts[1]
        product = ProductTerm()
        terms = []
        for i, bit in enumerate(inputs):
            if bit == '0':
                product.zeros.append(self.inputs[i])
                terms.append(Not(self.inputs[i]))
            elif bit == '1':
                product.ones.append(self.inputs[i])
                terms.append(self.inputs[i])
        product.expr = And(*terms)

        for i, bit in enumerate(outputs):
            if bit == '1':
                self.or_terms[self.outputs[i]].products.append(product)
                self.or_terms[self.outputs[i]].expr = Or(
                    self.or_terms[self.outputs[i]].expr, product.expr)

    def read_xor_term(self, line: str):
        parts = line.split()
        assert len(parts) == 2
        assert len(parts[0]) == len(self.inputs)
        assert len(parts[1]) == len(self.outputs)

        inputs = parts[0]
        outputs = parts[1]
        terms = []
        for i, bit in enumerate(inputs):
            if bit == '1':
                terms.append(self.inputs[i])
        for i, bit in enumerate(outputs):
            if bit == '1':
                self.or_terms[self.outputs[i]].expr = Xor(*terms)


class Fitter():
    inputs: List[str]
    or_terms: Dict[str, OrTerm]
    all_or_terms: Dict[str, Dict[str, OrTerm]]
    input_mcs: Dict[str, int]
    input_sigs: Dict[str, str]

    def __init__(self):
        self.device = None
        self.next_mc = 1

        self.inputs = []
        self.outputs = []
        # A map of symbolic output to OrTerm
        self.or_terms = {}
        # A map of block to map of MC to OrTerm
        self.all_or_terms = {}
        self.all_or_exprs = {}

        # A map of symbolic input to macrocell number
        self.input_mcs = {}
        # A map of symbolic input to multiplexer signal name
        self.input_sigs = {}

    def map_inputs(self):
        print("Mapping pin inputs")
        db = get_database()
        self.device = db["ATF1502AS"]

        # For now, assuming this is an input layer, map inputs directly onto
        # MCs starting with MC1. We can use an input MC as an intermediate
        # output by routing its output to MCn_FB.

        self.input_mcs = {top_input: self.get_next_mc()
                          for top_input in self.inputs}
        for top_input, input_mc in self.input_mcs.items():
            pin = self.device["pins"]["PLCC44"][f"M{input_mc}"]
            self.input_sigs = {top_input: f"M{input_mc}_PAD" for top_input,
                               input_mc in self.input_mcs.items()}
            print(f"assign input {top_input} to MC{input_mc} (pin {pin})")
            print(f"  set MC{input_mc}.oe_mux GND")

        # This isn't accurate. It's only accurate when the number of intermediate
        # outputs exceeds the number of inputs.
        self.next_mc = 1

        # Initialize blocks in all_or_terms
        for block in self.device["blocks"].keys():
            self.all_or_terms[block] = {}
            self.all_or_exprs[block] = {}

    def get_next_mc(self) -> int:
        specials = [4, 9, 25, 20]  # TDI, TMS, TCK, TDO
        if self.next_mc in specials:
            self.next_mc += 2
        elif self.next_mc > 32:
            return None
        else:
            self.next_mc += 1
        return self.next_mc-1

    def map_output_layer(self):
        device = self.device

        for i, output in enumerate(self.outputs):
            mc = self.input_mcs[output]
            pin = device["pins"]["PLCC44"][f"M{mc}"]
            print(f"Output {output} is at MC{mc} (pin {pin})")
            print(f"  set MC{mc}.o_mux comb")
            print(f"  set MC{mc}.oe_mux pt5")
            print(f"  set MC{mc}.pt5_func as")

    def map_and_or_layer(self):
        print("Mapping AND-OR layer")
        device = self.device

        # For now, map the outputs directly onto MCs starting with
        # MC1.
        for output in self.outputs:
            or_term = self.or_terms[output]
            or_expr = or_term.expr
            inv = False
            print(f"{output} = {or_term.expr}")
            if isinstance(or_expr, pyeda.boolalg.expr.OrOp) and len(or_expr.xs) > 5:
                # Maybe we can invert, and then use the macrocell's inverter to invert
                # the result?
                nor_expr = espresso_exprs(Not(or_term.expr).to_dnf())
                # espresso_expr returns a tuple
                # to_dnf converts an expression to disjunctive normal form
                # (i.e. sum of products).
                nor_expr = nor_expr[0].to_dnf()
                print(f"Try the inverse of this instead: {nor_expr}")
                if isinstance(nor_expr, pyeda.boolalg.expr.OrOp) and len(or_expr.xs) > 5:
                    print(
                        f"ERROR: or-term for {output} needs more than"
                        " one macrocell (5 products), which is not supported yet.")
                    return
                or_expr = nor_expr
                inv = True

            mc = self.get_next_mc()
            assert mc is not None, "Ran out of macrocells"
            mc_name = f"MC{mc}"
            macrocell = device["macrocells"][mc_name]
            block = macrocell["block"]
            print(f"output {output} mapped to {mc_name}.FB in block {block}")
            self.all_or_terms[block][mc_name] = or_term
            self.all_or_exprs[block][mc_name] = or_expr
            self.input_mcs[output] = mc
            self.input_sigs[output] = f"MC{mc}_FB"

            print(f"set {mc_name}.pt_power on")
            print(f"set {mc_name}.pt1_mux sum")
            print(f"set {mc_name}.pt2_mux sum")
            print(f"set {mc_name}.pt3_mux sum")
            print(f"set {mc_name}.pt4_mux sum")
            print(f"set {mc_name}.pt5_mux sum")
            print(f"set {mc_name}.fb_mux xt")
            print(f"set {mc_name}.xor_a_mux sum")
            print(f"set {mc_name}.xor_b_mux VCC_pt12")

            # It's weird, but because we have to feed a 1 into one input of
            # the macrocell's XOR, it naturally inverts. There's another
            # optional inverter after that, so if we want the non-inverted
            # output of the OR gate, we have to turn that inverter on!
            if inv:
                print(f"set {mc_name}.xor_invert off")
            else:
                print(f"set {mc_name}.xor_invert on")

        # Now that we've mapped inputs to outputs,
        # add them to the inputs and clear out the outputs.
        self.inputs += self.outputs
        self.outputs = []

        print("Input mcs:")
        pprint.pprint(self.input_mcs)
        print("Input sigs:")
        pprint.pprint(self.input_sigs)

    def map_and_xor_layer(self):
        print("Mapping XOR layer")
        device = self.device

        # For now, map the outputs directly onto MCs starting with
        # the next MC
        for output in self.outputs:
            expr = self.or_terms[output].expr
            assert isinstance(expr, pyeda.boolalg.expr.XorOp)
            if len(expr.xs) != 2:
                print(
                    f"ERROR: xor-term for {output} does not have 2 products, which is not supported yet.")
                return
            mc = self.get_next_mc()
            assert mc is not None, "Ran out of macrocells"
            mc_name = f"MC{mc}"
            macrocell = device["macrocells"][mc_name]
            block = macrocell["block"]
            print(f"output {output} mapped to {mc_name}.FB in block {block}")
            self.all_or_exprs[block][mc_name] = expr
            self.input_mcs[output] = mc
            self.input_sigs[output] = f"MC{mc}_FB"

            print(f"set {mc_name}.pt_power on")
            print(f"set {mc_name}.pt1_mux sum")
            print(f"set {mc_name}.pt2_mux xor")
            print(f"set {mc_name}.pt3_mux sum")
            print(f"set {mc_name}.pt4_mux sum")
            print(f"set {mc_name}.pt5_mux sum")
            print(f"set {mc_name}.fb_mux xt")
            print(f"set {mc_name}.xor_a_mux sum")
            print(f"set {mc_name}.xor_b_mux VCC_pt12")
            print(f"set {mc_name}.xor_invert on")

        # Now that we've mapped inputs to outputs,
        # add them to the inputs and clear out the outputs.
        self.inputs += self.outputs
        self.outputs = []

        print("Input mcs:")
        pprint.pprint(self.input_mcs)
        print("Input sigs:")
        pprint.pprint(self.input_sigs)

    def set_uims(self):
        # Collect all MCn_FB and Mn_PAD before choosing UIMs for each block.
        # This is an instance of the assignment problem, which we solve using the
        # Hungarian algorithm, which is O(n^3). The hope is that because the matrix
        # is extremely sparse, the algorithm runs very quickly.

        switches = self.device["switches"]

        # Map signals to UIMs, per block
        sig_to_uim = {}
        for blk in dev["blocks"].keys():
            sig_to_uim[blk] = {}
        for switch, data in switches.items():
            blk = data["block"]
            switch_sigs = data["mux"]["values"].keys()
            for sig in switch_sigs:
                if sig not in sig_to_uim[blk]:
                    sig_to_uim[blk][sig] = []
                sig_to_uim[blk][sig].append(switch)

        for blk in self.all_or_exprs:
            print(f"Constructing set of signals in block {blk}")
            # Construct the set of needed signals.
            sigs = set()
            for or_expr in self.all_or_exprs[blk].values():
                sigs.update(set(self.input_sigs[str(term)]
                                for term in or_expr.support))

            # Convert to ordered array
            sigs = [s for s in sigs]
            if len(sigs) == 0:
                print(f"No used signals in block {blk}")
                continue
            print(f"Used signals in block {blk}: {sigs}")

            # Construct the set of candidate switches for those signals.
            candidate_switches = set()
            for sig in sigs:
                candidate_switches.update(set(s for s in sig_to_uim[blk][sig]))
            # Convert to ordered array
            candidate_switches = [s for s in candidate_switches]
            print(f"Candidate switches in block {blk}: {candidate_switches}")

            # Construct the cost matrix. We assign an different cost per candidate
            # switch to help the algorithm be stable.
            matrix = [[DISALLOWED for _ in range(
                len(candidate_switches))] for _ in range(len(sigs))]
            for row, sig in enumerate(sigs):
                cost = 1
                for candidate_switch in sig_to_uim[blk][sig]:
                    col = candidate_switches.index(candidate_switch)
                    matrix[row][col] = cost
                    cost += 1
            cost_matrix = make_cost_matrix(
                matrix, lambda cost: cost if cost != DISALLOWED else DISALLOWED)

            # Assign signals to switches.
            m = Munkres()
            indexes = m.compute(cost_matrix)
            sig_to_switch = {}
            # print_matrix(matrix, 'Based on this matrix:')
            print("Setting UIM fuses:")
            for r, c in indexes:
                v = matrix[r][c]
                print(f"set {candidate_switches[c]} {sigs[r]}")
                sig_to_switch[sigs[r]] = candidate_switches[c]
            # pprint.pprint(sig_to_switch)

            print("Setting product term fuses:")
            for mc_name, or_expr in self.all_or_exprs[blk].items():
                products = or_expr.xs if isinstance(or_expr, pyeda.boolalg.expr.OrOp) or isinstance(
                    or_expr, pyeda.boolalg.expr.XorOp) else [or_expr]

                for ptn, product in enumerate(products):
                    terms = product.xs if isinstance(
                        product, pyeda.boolalg.expr.AndOp) else [product]
                    for sig in terms:
                        inv = isinstance(sig, pyeda.boolalg.expr.Complement)
                        sig = str(Not(sig) if inv else sig)
                        uim = sig_to_switch[self.input_sigs[sig]]
                        switch_polarity = "_N" if inv else "_P"
                        print(
                            f"      set {mc_name}.PT{ptn} +{uim}{switch_polarity}")


if __name__ == "__main__":
    db = get_database()
    dev = db["ATF1502AS"]

    parse = PLAParser(sys.argv[1])

    p = Fitter()
    p.inputs = parse.inputs
    p.outputs = parse.outputs
    p.or_terms = parse.or_terms

    p.map_inputs()

    for arg in sys.argv[1:]:
        parse = PLAParser(arg)
        p.outputs = parse.outputs
        p.or_terms = parse.or_terms

        if parse.is_xor:
            p.map_and_xor_layer()
        elif parse.is_outputs:
            p.map_output_layer()
        else:
            p.map_and_or_layer()

    p.set_uims()
