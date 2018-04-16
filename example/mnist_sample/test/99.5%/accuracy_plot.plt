set border lc rgb "black"
set grid lc rgb "#D8D8D8" lt 2
set key opaque box
set object 1 rect behind from screen 0,0 to screen 1,1 fc rgb "#FAFAFA" fillstyle solid
set key right bottom

# smooth [unique, csplines, acsplines, bezier, sbezier]

plot 'accuracy_rate.dat' using 1:2   t "varidation accuracy"  with lines linewidth 1 linecolor rgbcolor "#F5A9A9" dt 1

replot 'accuracy_rate.dat' using 1:3  t "test sample accuracy" with lines linewidth 1 linecolor rgbcolor "#A9BCF5" dt 1
#replot 'accuracy_gool.dat' t "gool" with lines linewidth 2 linecolor rgbcolor "green" dt 1


replot 'accuracy_rate.dat' using 1:2  smooth bezier t "varidation"  with lines linewidth 2 linecolor rgbcolor "red"
replot 'accuracy_rate.dat' using 1:3  smooth bezier t "test" with lines linewidth 2 linecolor rgbcolor "blue"

pause 10
reread
