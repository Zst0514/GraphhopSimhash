module hamming_threshold_compare #(
    parameter int WIDTH = 16,
    parameter int THRESHOLD = 2,
    parameter int COUNT_W = $clog2(WIDTH + 1)
) (
    input  logic             clk,
    input  logic [WIDTH-1:0] query,
    input  logic [WIDTH-1:0] stored,
    output logic             match
);
    logic [COUNT_W-1:0] dist;
    integer idx;

    always_comb begin
        dist = '0;
        for (idx = 0; idx < WIDTH; idx = idx + 1) begin
            dist = dist + (query[idx] ^ stored[idx]);
        end
    end

    always_ff @(posedge clk) begin
        match <= (dist <= THRESHOLD);
    end
endmodule

module hamming_threshold_compare16 (
    input  logic        clk,
    input  logic [15:0] query,
    input  logic [15:0] stored,
    output logic        match
);
    hamming_threshold_compare #(
        .WIDTH(16),
        .THRESHOLD(2)
    ) u_compare (
        .clk(clk),
        .query(query),
        .stored(stored),
        .match(match)
    );
endmodule

module hamming_threshold_compare64 (
    input  logic        clk,
    input  logic [63:0] query,
    input  logic [63:0] stored,
    output logic        match
);
    hamming_threshold_compare #(
        .WIDTH(64),
        .THRESHOLD(2)
    ) u_compare (
        .clk(clk),
        .query(query),
        .stored(stored),
        .match(match)
    );
endmodule
