set border lc rgb "black"
set grid lc rgb "#D8D8D8" lt 2
set key opaque box
set object 1 rect behind from screen 0,0 to screen 1,1 fc rgb "#FAFAFA" fillstyle solid


plot 'error_loss.dat' using 1:2   t "varidation"  with lines linewidth 1 linecolor rgbcolor "#F5A9A9" dt 1
replot 'error_loss.dat' using 1:3  t "test sample" with lines linewidth 1 linecolor rgbcolor "#A9BCF5" dt 1
#replot 'error_loss_gool.dat'  t "gool" with lines linewidth 2 linecolor rgbcolor "green" dt 1


replot 'error_loss.dat' using 1:2  smooth bezier t "varidation"  with lines linewidth 2 linecolor rgbcolor "red"
replot 'error_loss.dat' using 1:3  smooth bezier t "test" with lines linewidth 2 linecolor rgbcolor "blue"

pause 10
reread
