module hamming_threshold_compare #(parameter WIDTH = 16, parameter THRESHOLD = 2, parameter COUNT_W = 5) (
    clk,
    query,
    stored,
    match
);
    input clk;
    input [WIDTH-1:0] query;
    input [WIDTH-1:0] stored;
    output match;

    reg match;
    reg [COUNT_W-1:0] dist;
    integer idx;

    always @* begin
        dist = {COUNT_W{1'b0}};
        for (idx = 0; idx < WIDTH; idx = idx + 1) begin
            dist = dist + (query[idx] ^ stored[idx]);
        end
    end

    always @(posedge clk) begin
        match <= (dist <= THRESHOLD);
    end
endmodule

module hamming_threshold_compare16 (
    clk,
    query,
    stored,
    match
);
    input clk;
    input [15:0] query;
    input [15:0] stored;
    output match;

    hamming_threshold_compare #(
        .WIDTH(16),
        .THRESHOLD(2),
        .COUNT_W(5)
    ) u_compare (
        .clk(clk),
        .query(query),
        .stored(stored),
        .match(match)
    );
endmodule

module hamming_threshold_compare64 (
    clk,
    query,
    stored,
    match
);
    input clk;
    input [63:0] query;
    input [63:0] stored;
    output match;

    hamming_threshold_compare #(
        .WIDTH(64),
        .THRESHOLD(2),
        .COUNT_W(7)
    ) u_compare (
        .clk(clk),
        .query(query),
        .stored(stored),
        .match(match)
    );
endmodule
