# Fixed-Point LIF Neuron — HDL Specification

**Generated:** 2026-07-20
**Target:** PMUT Neuromorphic Classification Pipeline
**Neuron Model:** Leaky Integrate-and-Fire (LIF)

---

## 1. Module Overview

### Module Name
`lif_neuron_fxp`

### Purpose
Single LIF neuron operating in fixed-point Q3.12 arithmetic.
Accepts membrane potential, input current, and configuration parameters.
Produces updated membrane potential and spike output.

### Ports

| Port | Direction | Width | Description |
|---|---|---|---|
| `clk` | input | 1 | System clock |
| `rst_n` | input | 1 | Active-low reset |
| `i_current` | input | `TOTAL_BITS` | Input synaptic current (Q3.12) |
| `i_beta` | input | `TOTAL_BITS` | Membrane decay factor β (Q3.12) |
| `i_threshold` | input | `TOTAL_BITS` | Firing threshold θ (Q3.12) |
| `i_membrane_init` | input | `TOTAL_BITS` | Initial membrane potential |
| `i_valid` | input | 1 | Input valid strobe |
| `o_spike` | output | 1 | Spike output (1 = fire) |
| `o_membrane` | output | `TOTAL_BITS` | Updated membrane potential (Q3.12) |
| `o_ready` | output | 1 | Output valid strobe |

---

## 2. Fixed-Point Format

**Q3.12 format:**
- Total bits: 16 (1 sign + 3 integer + 12 fractional)
- Range: [-8.0, 7.999755859375]
- Resolution (ε): 0.000244140625
- Scale factor: 2^12 = 4096

### Fixed-Point Arithmetic Rules

1. **Addition:** Aligned Q-format → direct integer addition with overflow saturation.
2. **Multiplication:** (Qa × Qb) >> 12 → truncate to Q3.12.
3. **Comparison:** Direct integer comparison (same Q-format → same scale).
4. **Overflow:** Saturate to `MAX_POS` / `MAX_NEG` (no wrap-around).

---

## 3. LIF Neuron Update Rule

### Mathematical Definition (Fixed-Point Integer)

```
I_q     = i_current                                           // synaptic input
V_q     = ((i_beta * i_membrane) >> FRAC_BITS) + I_q          // leak + integrate
V_q     = CLAMP(V_q, MIN_VAL, MAX_VAL)                       // saturation

if (V_q >= i_threshold) begin
    SPIKE   = 1
    V_q     = V_q - i_threshold                                // subtractive reset
    V_q     = CLAMP(V_q, MIN_VAL, MAX_VAL)
end else begin
    SPIKE   = 0
end
```

### Verilog Implementation (Behavioral)

```verilog
module lif_neuron_fxp #(
    parameter INTEGER_BITS  = 3,
    parameter FRACTIONAL_BITS = 12,
    parameter TOTAL_BITS    = 16,
    parameter MAX_VAL       = 32767,
    parameter MIN_VAL       = -32768
) (
    input  wire                      clk,
    input  wire                      rst_n,
    input  wire [TOTAL_BITS-1:0]     i_current,
    input  wire [TOTAL_BITS-1:0]     i_beta,
    input  wire [TOTAL_BITS-1:0]     i_threshold,
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
                    // 1. Leak: i_beta * membrane
                    mult_temp = $signed(i_beta) * $signed(membrane);
                    // Truncate fractional bits
                    membrane <= (mult_temp >>> FRACTIONAL_BITS) + $signed(i_current);

                    // 2. Overflow saturation
                    if ($signed(membrane) > $signed(MAX_VAL))
                        membrane <= MAX_VAL;
                    else if ($signed(membrane) < $signed(MIN_VAL))
                        membrane <= MIN_VAL;

                    // 3. Spike + reset
                    if ($signed(membrane) >= $signed(i_threshold)) begin
                        o_spike <= 1'b1;
                        membrane <= $signed(membrane) - $signed(i_threshold);
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
| Q-format | Q3.12 (16-bit signed) |
| β (membrane decay) | 3686 (quantized from 0.9) |
| θ (threshold) | 4096 (quantized from 1.0) |
| N_NEURONS | 10 |
| N_INPUT | 4 |

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
