# trying-using-tiny-dnn

## Code Examples

input image 32x32x1 channle 
tiny-dnn (original)

```cpp
  nn << conv(32, 32, 5, 1, 6, padding::valid, true, 1, 1, backend_type)
     << tanh()
     << ave_pool(28, 28, 6, 2)
     << tanh()
     << conv(14, 14, 5, 6, 16, connection_table(tbl, 6, 16), padding::valid, true, 1, 1, backend_type)
     << tanh()
     << ave_pool(10, 10, 16, 2)
     << tanh()
     << conv(5, 5, 5, 16, 120, padding::valid, true, 1, 1, backend_type)
     << tanh()
     << fc(120, 10, true, backend_type)
     << tanh();
```
conv(32, 32, 5, 1, 6...) -> tanh() -> ave_pool(**28**, **28**, 6, 2) 
**28** = (32 - 5 + 1) / stride 
**This calculation is troublesome!! **

**I tried not to calculate the output size ** 
```cpp
 	LayerInfo layers(in_w, in_h, in_map);
	nn << layers.add_cnv(6, 5, 1, padding::valid);
	nn << tanh();
	nn << layers.add_avepool(2, 2);
	nn << tanh();
	nn << layers.add_cnv(16, 5, 1, padding::valid, true, connection_table(tbl, 6, 16));
	nn << tanh();
	nn << layers.add_avepool(2, 2);
	nn << tanh();
	nn << layers.add_cnv(120, 5, 1, padding::valid);
	nn << tanh();
	nn << layers.add_fc(10);
	nn << tanh();;

