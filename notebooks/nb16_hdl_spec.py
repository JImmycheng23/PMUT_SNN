"""
Notebook 16: HDL-Ready Specification and Optional Verilog LIF
==============================================================

Writes the complete HDL specification for a fixed-point LIF neuron
and optionally generates synthesis-ready Verilog.

Usage:
  python notebooks/nb16_hdl_spec.py                 # Write specification
  python notebooks/nb16_hdl_spec.py --generate-hdl  # Also generate Verilog
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "hdl"


# ---------------------------------------------------------------------------
# HDL Specification (Markdown)
# ---------------------------------------------------------------------------

HDL_SPEC = """# Fixed-Point LIF Neuron — HDL Specification

**Generated:** $$DATE$$
**Target:** PMUT Neuromorphic Classification Pipeline
**Neuron Model:** Leaky Integrate-and-Fire (LIF)
**Scope:** Architecture-level specification — defines the generic LIF neuron building block.
This spec is NOT derived from trained classifier weights; it describes the neuron
hardware primitive. PMUT-derived test vectors (from NB15) are used for verification.

---

## 1. Module Overview

### Module Name
`lif_neuron_fxp`

### Purpose
Single LIF neuron operating in fixed-point Q$$INT_BITS$$.$$FRAC_BITS$$ arithmetic.
Accepts membrane potential, input current, and configuration parameters.
Produces updated membrane potential and spike output.

### Ports

| Port | Direction | Width | Description |
|---|---|---|---|
| `clk` | input | 1 | System clock |
| `rst_n` | input | 1 | Active-low reset |
| `i_current` | input | `TOTAL_BITS` | Input synaptic current (Q$$INT_BITS$$.$$FRAC_BITS$$) |
| `i_valid` | input | 1 | Input valid strobe |
| `o_spike` | output | 1 | Spike output (1 = fire) |
| `o_membrane` | output | `TOTAL_BITS` | Updated membrane potential (Q$$INT_BITS$$.$$FRAC_BITS$$) |
| `o_ready` | output | 1 | Output valid strobe |

**Parameters** (hardcoded as Verilog `parameter`, not runtime inputs):

| Parameter | Value | Description |
|---|---|---|
| `BETA` | $$BETA_Q$$ | Membrane decay β = 0.9 → $$BETA_FLOAT$$ in Q$$INT_BITS$$.$$FRAC_BITS$$ |
| `THRESHOLD` | $$THRESHOLD_Q$$ | Firing threshold θ = 1.0 → $$THRESHOLD_FLOAT$$ in Q$$INT_BITS$$.$$FRAC_BITS$$ |
| `MAX_VAL` | $$MAX_VAL_Q$$ | Positive saturation limit |
| `MIN_VAL` | $$MIN_VAL_Q$$ | Negative saturation limit |

β and threshold are parameters (not runtime inputs) because they are constants for a given neuron configuration. The Q-format values are computed at elaboration time.

---

## 2. Fixed-Point Format

**Q$$INT_BITS$$.$$FRAC_BITS$$ format:**
- Total bits: $$TOTAL_BITS$$ (1 sign + $$INT_BITS$$ integer + $$FRAC_BITS$$ fractional)
- Range: [$$MIN_VAL$$, $$MAX_VAL$$]
- Resolution (ε): $$EPSILON$$
- Scale factor: 2^$$FRAC_BITS$$ = $$SCALE$$

### Fixed-Point Arithmetic Rules

1. **Addition:** Aligned Q-format → direct integer addition with overflow saturation.
2. **Multiplication:** (Qa × Qb) >> $$FRAC_BITS$$ → truncate to Q$$INT_BITS$$.$$FRAC_BITS$$.
3. **Comparison:** Direct integer comparison (same Q-format → same scale).
4. **Overflow:** Saturate to `MAX_POS` / `MAX_NEG` (no wrap-around).

---

## 3. LIF Neuron Update Rule

### Mathematical Definition (Fixed-Point Integer)

```
// BETA and THRESHOLD are module parameters (not runtime inputs)
I_q     = i_current                                           // synaptic input
V_q     = ((BETA * membrane) >> FRAC_BITS) + I_q             // leak + integrate
V_q     = CLAMP(V_q, MIN_VAL, MAX_VAL)                       // saturation

if (V_q >= THRESHOLD) begin
    SPIKE   = 1
    V_q     = V_q - THRESHOLD                                  // subtractive reset
    V_q     = CLAMP(V_q, MIN_VAL, MAX_VAL)
end else begin
    SPIKE   = 0
end
```

### Verilog Implementation (Behavioral)

```verilog
module lif_neuron_fxp #(
    parameter INTEGER_BITS  = $$INT_BITS$$,
    parameter FRACTIONAL_BITS = $$FRAC_BITS$$,
    parameter TOTAL_BITS    = $$TOTAL_BITS$$,
    parameter MAX_VAL       = $$MAX_VAL_Q$$,
    parameter MIN_VAL       = $$MIN_VAL_Q$$,
    parameter THRESHOLD     = $$THRESHOLD_Q$$,
    parameter BETA          = $$BETA_Q$$
) (
    input  wire                      clk,
    input  wire                      rst_n,
    input  wire [TOTAL_BITS-1:0]     i_current,
    input  wire                      i_valid,
    output reg                       o_spike,
    output reg  [TOTAL_BITS-1:0]     o_membrane,
    output reg                       o_ready
);

    // Internal registers
    reg [TOTAL_BITS-1:0]  membrane;
    reg [2*TOTAL_BITS-1:0] mult_temp;  // Double-width for multiplication

    // State machine
    localparam IDLE = 2'b00, COMPUTE = 2'b01, DONE = 2'b10;
    reg [1:0] state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            membrane <= {TOTAL_BITS{{1'b0}}};
            o_spike  <= 1'b0;
            o_membrane <= {TOTAL_BITS{{1'b0}}};
            o_ready  <= 1'b0;
            state    <= IDLE;
        end else begin
            case (state)
                IDLE: begin
                    o_ready <= 1'b0;
                    if (i_valid) begin
                        state <= COMPUTE;
                    end
                end

                COMPUTE: begin
                    // 1. Leak: BETA * membrane (fixed-point multiply)
                    mult_temp = $signed(BETA) * $signed(membrane);
                    // Truncate fractional bits, add input current
                    membrane <= (mult_temp >>> FRACTIONAL_BITS) + $signed(i_current);

                    // 2. Overflow saturation
                    if ($signed(membrane) > $signed(MAX_VAL))
                        membrane <= MAX_VAL;
                    else if ($signed(membrane) < $signed(MIN_VAL))
                        membrane <= MIN_VAL;

                    // 3. Spike + subtractive reset
                    if ($signed(membrane) >= $signed(THRESHOLD)) begin
                        o_spike <= 1'b1;
                        membrane <= $signed(membrane) - $signed(THRESHOLD);
                    end else begin
                        o_spike <= 1'b0;
                    end

                    o_membrane <= membrane;
                    o_ready <= 1'b1;
                    state <= DONE;
                end

                DONE: begin
                    o_ready <= 1'b0;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
```

---

## 4. Multi-Neuron Architecture

For an N-neuron reservoir, instantiate N `lif_neuron_fxp` instances with
shared input bus and individual weight ROMs.

```
                 ┌─────────────────┐
  spike_in[3:0] ─┤ Weight ROM ×4   │── current[9:0]
                 └─────────────────┘        │
                                     ┌──────▼──────┐
                                     │ LIF Neuron 0 │── spike[0]
                                     └──────────────┘
                                     ┌──────▼──────┐
                                     │ LIF Neuron 1 │── spike[1]
                                     └──────────────┘
                                            ...
                                     ┌──────▼──────┐
                                     │ LIF Neuron N │── spike[N-1]
                                     └──────────────┘
```

---

## 5. Testbench Requirements

### Input Signals
- PMUT-derived spike train from `pmut_input_spikes.csv` (NB15 output)
- Q-format parameters from `params.json`

### Verification Checklist

| Check | Method |
|---|---|
| Membrane trace matches Python reference | Compare `expected_voltage_q.csv` bit-by-bit |
| Spike output matches Python reference | Compare `expected_spikes.csv` |
| No overflow/underflow without saturation | Monitor internal signals |
| Reset behavior correct | Toggle rst_n mid-sequence |
| Pipeline timing met | Static timing analysis |

### Testbench Pseudocode

```verilog
// Read pmut_input_spikes.csv
// Read expected_spikes.csv
// for each time step:
//     apply input spike → run lif_neuron_fxp
//     assert(o_spike == expected_spike[t])
//     assert(o_membrane == expected_voltage[t] within 1 LSB)
```

---

## 6. Synthesis Notes

- **Target clock:** 50–100 MHz (typical FPGA)
- **Critical path:** Multiplier (beta × membrane) → adder → comparator
- **Latency:** 2–3 clock cycles per time step (pipelineable)
- **Resource estimate (per neuron):**
  - 1 × signed multiplier (TOTAL_BITS × TOTAL_BITS)
  - 1 × signed adder
  - 1 × signed comparator
  - ~ (2 × TOTAL_BITS + 1) flip-flops
- **N-neuron reservoir:** N × multiplier + N × adder (parallel)

---

## 7. Parameter Defaults

| Parameter | Value |
|---|---|
| Q-format | Q$$INT_BITS$$.$$FRAC_BITS$$ ($$TOTAL_BITS$$-bit signed) |
| β (membrane decay) | $$BETA_Q$$ (quantized from $$BETA_FLOAT$$) |
| θ (threshold) | $$THRESHOLD_Q$$ (quantized from $$THRESHOLD_FLOAT$$) |
| N_NEURONS | $$N_NEURONS$$ |
| N_INPUT | $$N_INPUT$$ |

---

## 8. Verification Status

- [x] Python floating-point LIF reference model (NB7)
- [x] Python fixed-point LIF model (NB13)
- [x] Bit-width sensitivity study — 16-bit recommended (NB13)
- [x] PMUT inference at fixed-point (NB14)
- [x] HDL test vectors generated (NB15)
- [ ] HDL testbench simulation
- [ ] FPGA synthesis and timing closure

---

*Specification prepared for PMUT Neuromorphic Classification Project, Goal 4.*
"""


def generate_spec(qfmt_bits: int = 16, n_neurons: int = 10, n_input: int = 4) -> str:
    """Fill in the spec template with actual Q-format values."""
    from src.pmut_models.fixed_point import QFormat, float_to_fixed

    if qfmt_bits == 16:
        qfmt = QFormat(3, 12)
    elif qfmt_bits == 12:
        qfmt = QFormat(3, 8)
    else:
        qfmt = QFormat(3, 12)

    beta_float = 0.9
    threshold_float = 1.0

    # Use $$KEY$$ placeholders to avoid conflict with Verilog {braces}
    replacements = {
        "$$DATE$$": datetime.now().strftime("%Y-%m-%d"),
        "$$INT_BITS$$": str(qfmt.integer_bits),
        "$$FRAC_BITS$$": str(qfmt.fractional_bits),
        "$$TOTAL_BITS$$": str(qfmt.total_bits),
        "$$MIN_VAL$$": str(qfmt.min_val),
        "$$MAX_VAL$$": str(qfmt.max_val),
        "$$EPSILON$$": str(qfmt.epsilon),
        "$$SCALE$$": str(int(qfmt.scale)),
        "$$MAX_VAL_Q$$": str(int(qfmt.max_val * qfmt.scale)),
        "$$MIN_VAL_Q$$": str(int(qfmt.min_val * qfmt.scale)),
        "$$BETA_Q$$": str(int(float_to_fixed(beta_float, qfmt))),
        "$$BETA_FLOAT$$": str(beta_float),
        "$$THRESHOLD_Q$$": str(int(float_to_fixed(threshold_float, qfmt))),
        "$$THRESHOLD_FLOAT$$": str(threshold_float),
        "$$N_NEURONS$$": str(n_neurons),
        "$$N_INPUT$$": str(n_input),
    }

    spec = HDL_SPEC
    for old, new in replacements.items():
        spec = spec.replace(old, new)
    return spec


def generate_verilog(qfmt_bits: int = 16) -> str:
    """Generate standalone Verilog LIF neuron module."""
    from src.pmut_models.fixed_point import QFormat, float_to_fixed

    if qfmt_bits == 16:
        qfmt = QFormat(3, 12)
    else:
        qfmt = QFormat(3, 12)

    beta_q = int(float_to_fixed(0.9, qfmt))
    threshold_q = int(float_to_fixed(1.0, qfmt))

    TOTAL = qfmt.total_bits
    MAX_VAL_Q = int(qfmt.max_val * qfmt.scale)
    MIN_VAL_Q = int(qfmt.min_val * qfmt.scale)

    return f"""// Fixed-Point LIF Neuron — Synthesis-Ready Verilog
// Q{qfmt.integer_bits}.{qfmt.fractional_bits} format ({TOTAL}-bit signed)
// Auto-generated by NB16 for PMUT Neuromorphic Project

module lif_neuron_fxp #(
    parameter INTEGER_BITS  = {qfmt.integer_bits},
    parameter FRACTIONAL_BITS = {qfmt.fractional_bits},
    parameter TOTAL_BITS    = {TOTAL},
    parameter MAX_VAL       = {TOTAL}'d{MAX_VAL_Q},
    parameter MIN_VAL       = -{TOTAL}'d{abs(MIN_VAL_Q)},
    parameter THRESHOLD     = {TOTAL}'d{threshold_q},
    parameter BETA          = {TOTAL}'d{beta_q}
) (
    input  wire                      clk,
    input  wire                      rst_n,
    input  wire [TOTAL_BITS-1:0]     i_current,
    input  wire                      i_valid,
    output reg                       o_spike,
    output reg  [TOTAL_BITS-1:0]     o_membrane,
    output reg                       o_ready
);

    reg [TOTAL_BITS-1:0]   membrane;
    reg [2*TOTAL_BITS-1:0] mult_temp;

    localparam IDLE = 2'b00, COMPUTE = 2'b01, DONE = 2'b10;
    reg [1:0] state;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            membrane  <= {{TOTAL_BITS}}{{1'b0}};
            o_spike   <= 1'b0;
            o_membrane <= {{TOTAL_BITS}}{{1'b0}};
            o_ready   <= 1'b0;
            state     <= IDLE;
        end else begin
            case (state)
                IDLE: begin
                    o_ready <= 1'b0;
                    if (i_valid) state <= COMPUTE;
                end

                COMPUTE: begin
                    // Leak: beta * membrane (fixed-point multiply)
                    mult_temp = $signed(BETA) * $signed(membrane);
                    membrane <= (mult_temp >>> FRACTIONAL_BITS) + $signed(i_current);

                    // Saturation
                    if ($signed(membrane) > $signed(MAX_VAL))
                        membrane <= MAX_VAL;
                    else if ($signed(membrane) < $signed(MIN_VAL))
                        membrane <= MIN_VAL;

                    // Spike
                    if ($signed(membrane) >= $signed(THRESHOLD)) begin
                        o_spike <= 1'b1;
                        membrane <= $signed(membrane) - $signed(THRESHOLD);
                    end else begin
                        o_spike <= 1'b0;
                    end

                    o_membrane <= membrane;
                    o_ready <= 1'b1;
                    state <= DONE;
                end

                DONE: begin
                    o_ready <= 1'b0;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule


// Simple testbench
module lif_neuron_fxp_tb;
    reg clk, rst_n, i_valid;
    reg [{TOTAL}-1:0] i_current;
    wire o_spike, o_ready;
    wire [{TOTAL}-1:0] o_membrane;

    lif_neuron_fxp #(
        .INTEGER_BITS({qfmt.integer_bits}),
        .FRACTIONAL_BITS({qfmt.fractional_bits}),
        .TOTAL_BITS({TOTAL}),
        .MAX_VAL({TOTAL}'d{MAX_VAL_Q}),
        .MIN_VAL(-{TOTAL}'d{abs(MIN_VAL_Q)}),
        .THRESHOLD({TOTAL}'d{threshold_q}),
        .BETA({TOTAL}'d{beta_q})
    ) uut (
        .clk(clk), .rst_n(rst_n),
        .i_current(i_current), .i_valid(i_valid),
        .o_spike(o_spike), .o_membrane(o_membrane), .o_ready(o_ready)
    );

    // Clock generation
    always #5 clk = ~clk;  // 100 MHz

    initial begin
        $dumpfile(\"lif_neuron_fxp_tb.vcd\");
        $dumpvars(0, lif_neuron_fxp_tb);

        clk = 0; rst_n = 0; i_valid = 0; i_current = 0;
        #20 rst_n = 1;

        // Apply PMUT-derived test vectors (from NB15)
        // $readmemb(\"pmut_input_spikes.csv\", ...);
        // Compare o_spike with expected_spikes.csv

        #1000 $finish;
    end
endmodule
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="HDL Specification — NB16")
    parser.add_argument("--generate-hdl", action="store_true", help="Also generate Verilog")
    parser.add_argument("--qfmt-bits", type=int, default=16, choices=[16, 12])
    parser.add_argument("--n-neurons", type=int, default=10)
    parser.add_argument("--n-input", type=int, default=4)
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"NB16: HDL-Ready Specification")
    print(f"Q-format: {args.qfmt_bits}-bit  |  Neurons: {args.n_neurons}")
    print(f"{'='*60}")

    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate specification
    print("\n[1] Writing HDL specification...")
    spec = generate_spec(qfmt_bits=args.qfmt_bits, n_neurons=args.n_neurons, n_input=args.n_input)
    spec_path = out_dir / "HDL_SPECIFICATION.md"
    spec_path.write_text(spec, encoding="utf-8")
    print(f"  Specification → {spec_path}")

    # Generate Verilog (optional)
    if args.generate_hdl:
        print("\n[2] Generating Verilog...")
        verilog = generate_verilog(qfmt_bits=args.qfmt_bits)
        vlog_path = out_dir / "lif_neuron_fxp.v"
        vlog_path.write_text(verilog, encoding="utf-8")
        print(f"  Verilog → {vlog_path}")

    # Write simple testbench separately for the 16-bit case
    from src.pmut_models.fixed_point import QFormat, float_to_fixed
    qfmt = QFormat(3, 12) if args.qfmt_bits == 16 else QFormat(3, 8)

    print(f"\n{'='*60}")
    print(f"NB16 complete")
    print(f"\n  HDL Specification Contents:")
    print(f"    - Module ports (clk, rst_n, i_current, o_spike, o_membrane)")
    print(f"    - Q{qfmt.integer_bits}.{qfmt.fractional_bits} fixed-point format (range [{qfmt.min_val}, {qfmt.max_val}])")
    print(f"    - LIF update rule in integer arithmetic")
    print(f"    - Overflow/saturation handling")
    print(f"    - Multi-neuron reservoir architecture diagram")
    print(f"    - Testbench requirements and verification checklist")
    print(f"    - Synthesis notes (timing, resources, critical path)")
    print(f"  Outputs → {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
