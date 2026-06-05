set top_name [getenv "TOP_NAME"]
if {$top_name eq ""} {
    set top_name "hamming_threshold_compare16"
}

set clk_period_ns [getenv "CLK_PERIOD_NS"]
if {$clk_period_ns eq ""} {
    set clk_period_ns "2.0"
}

set lib_db [getenv "LIB_DB"]
if {$lib_db eq ""} {
    set lib_db "/pdk/tsmc28/logic/db/tcbn28hpcplusbwp40p140lvtssg0p9v125c_ccs.db"
}

set work_dir [pwd]
set rtl_dir [file normalize [file join $work_dir rtl]]
set rpt_dir [file normalize [file join $work_dir reports]]
set netlist_dir [file normalize [file join $work_dir netlists]]
file mkdir $rpt_dir
file mkdir $netlist_dir

set_app_var search_path [list $rtl_dir [file dirname $lib_db] $work_dir]
set_app_var target_library [list $lib_db]
set_app_var link_library "* $lib_db"

analyze -format verilog [file join $rtl_dir hamming_threshold_compare.v]
elaborate $top_name
current_design $top_name
link

create_clock -name clk -period $clk_period_ns [get_ports clk]
set_clock_uncertainty 0.05 [get_clocks clk]

set data_inputs [remove_from_collection [all_inputs] [get_ports clk]]
set_input_delay 0.05 -clock clk $data_inputs
set_input_transition 0.05 $data_inputs
set_output_delay 0.05 -clock clk [all_outputs]
set_load 0.005 [all_outputs]

compile_ultra

report_qor > [file join $rpt_dir ${top_name}.qor.rpt]
report_area > [file join $rpt_dir ${top_name}.area.rpt]
report_power > [file join $rpt_dir ${top_name}.power.rpt]
report_timing -delay max -max_paths 5 -path full_clock > [file join $rpt_dir ${top_name}.timing.rpt]

write -hierarchy -format verilog -output [file join $netlist_dir ${top_name}.mapped.v]
write_file -format ddc -hierarchy -output [file join $netlist_dir ${top_name}.ddc]

quit
