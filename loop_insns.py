"""loop_insns.py DISASSEMBLY SYMBOL FMAS_PER_ITERATION

The instructions one source iteration of SYMBOL's loop retires, counted in a disassembly listing of the compiled
binary: GNU objdump, llvm-objdump, otool -tv or dumpbin /disasm. Prints the count on the last line.

A loop is a backward branch inside the function: the instructions from its target to the branch, inclusive. The hot
loop is the one holding the most fused multiply-adds (an unroller's remainder loop holds fewer). A compiler that
unrolls U source iterations into one machine iteration puts U times the source's fused multiply-adds into it, so
U = (fused multiply-adds in the loop) / FMAS_PER_ITERATION, and one source iteration retires (loop instructions) / U.
"""
import re
import sys

ADDRESS = re.compile(r"^\s*(?:0x)?([0-9a-fA-F]+):?\s+(.*)$")
# A function's label in each format: "ADDRESS <sym>:" (objdump), "_sym:" (otool, Mach-O), "sym:" (dumpbin).
LABEL = re.compile(r"^\s*(?:[0-9a-fA-F]+\s+)?<?([A-Za-z_.$?@][\w.$?@]*)>?:\s*$")
FUSED = re.compile(r"^v?fn?m(add|sub|la|ls|acc|sac)", re.IGNORECASE)
BRANCH = re.compile(r"^(j[a-z]*|b(\.[a-z]+)?|b[a-z]{1,4}|cbn?z|tbn?z)$", re.IGNORECASE)
TARGET = re.compile(r"(?:0x)?([0-9a-fA-F]+)h?(?:\s*<[^>]*>)?\s*$")
# A trailing comment: "// b.any" (GNU objdump, AArch64), "# 4010 <sym>" (GNU objdump, x86), "; ..." (others).
COMMENT = re.compile(r"\s+(//|;|# ).*$")


def function_lines(text, symbol):
    lines, inside = [], False
    for raw in text.splitlines():
        label = LABEL.match(raw)
        if label:
            if inside:
                break  # the next function
            inside = label.group(1) in (symbol, "_" + symbol)
            continue
        match = ADDRESS.match(raw) if inside else None
        if match:
            lines.append((int(match.group(1), 16), match.group(2).strip()))
    return lines


def mnemonic_and_operands(text):
    # objdump and otool separate by tabs; dumpbin and llvm-objdump by runs of spaces; raw bytes are not printed
    # (--no-show-raw-insn, /disasm:nobytes).
    parts = COMMENT.sub("", text).split(None, 1)
    return (parts[0], parts[1] if len(parts) > 1 else "") if parts else ("", "")


def main():
    path, symbol, per_iteration = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with open(path, encoding="utf-8", errors="replace") as f:
        body = function_lines(f.read(), symbol)
    if not body:
        sys.exit(f"loop_insns: no instructions under {symbol} in {path}")
    start = body[0][0]
    loops = []
    for index, (address, text) in enumerate(body):
        mnemonic, operands = mnemonic_and_operands(text)
        target = TARGET.search(operands)
        if not BRANCH.match(mnemonic) or not target:
            continue
        to = int(target.group(1), 16)
        if start <= to <= address:
            inner = [t for a, t in body if to <= a <= address]
            fused = sum(1 for t in inner if FUSED.match(mnemonic_and_operands(t)[0]))
            loops.append((fused, -len(inner), to, address, inner))
    for fused, negative, to, address, inner in sorted(loops, reverse=True):
        print(f"loop {to:#x}..{address:#x}: {-negative} instructions, {fused} fused multiply-adds")
    if not loops or loops[0][0] == 0:
        sys.exit(f"loop_insns: no loop with a fused multiply-add in {symbol}")
    fused, negative, to, address, inner = max(loops)
    for text in inner:
        print("  " + text)
    if fused % per_iteration:
        sys.exit(f"loop_insns: {fused} fused multiply-adds is no multiple of {per_iteration} per iteration")
    unroll = fused // per_iteration
    print(f"hot loop {-negative} instructions, unrolled {unroll}x: instructions per iteration")
    print(f"{-negative / unroll:.4f}")


if __name__ == "__main__":
    main()
