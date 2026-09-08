"""Read a diagnostic Talos v7 image without consuming the active camera's triple buffer."""
import struct,pathlib,sys,time
from PIL import Image
meta=pathlib.Path('/tmp/talos_ipc_meta')
for attempt in range(8):
 m=meta.read_bytes()
 if struct.unpack_from('<II',m,0)!=(0x54414c07,7):raise RuntimeError('Expected Talos v7')
 slots=[struct.unpack_from('<QQIIBB',m,128+i*24768) for i in range(3)]
 slot=max(slots,key=lambda s:s[0]);seq,_,w,h,index,fmt=slot
 if not(0<w<=8192 and 0<h<=8192 and 0<=index<3 and fmt in (0,1)):continue
 with open('/tmp/talos_ipc_image_pool','rb') as f:f.seek(index*w*h*3);data=f.read(w*h*3)
 # Debug snapshot only; reject changed metadata and never write consumer indices.
 if m!=meta.read_bytes():continue
 Image.frombytes('RGB',(w,h),data,'raw','RGB' if fmt==0 else 'BGR').save(sys.argv[1]);print('frame',seq,'image',w,h);break
else:raise RuntimeError('No stable diagnostic frame; retry')
