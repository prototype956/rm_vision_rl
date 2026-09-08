"""Bounded read-only Foxglove JSON subscription; never starts MCAP recording."""
import argparse,base64,json,os,socket,struct,time
p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=30);p.add_argument('--output',required=True);args=p.parse_args()
s=socket.create_connection(('127.0.0.1',8765),timeout=5)
key=base64.b64encode(os.urandom(16)).decode()
s.sendall(('GET / HTTP/1.1\r\nHost: localhost:8765\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: '+key+'\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Protocol: foxglove.sdk.v1\r\n\r\n').encode())
buffer=b''
while b'\r\n\r\n' not in buffer:buffer+=s.recv(4096)
header,buffer=buffer.split(b'\r\n\r\n',1)
if header.split(b'\r\n',1)[0].split()[1]!=b'101':raise RuntimeError('WebSocket upgrade failed: '+header.split(b'\r\n',1)[0].decode())
def read(n):
 global buffer
 while len(buffer)<n:
  chunk=s.recv(max(4096,n-len(buffer)))
  if not chunk:raise EOFError
  buffer+=chunk
 out,buffer=buffer[:n],buffer[n:];return out
def send(payload,opcode=1):
 mask=os.urandom(4);n=len(payload)
 length=bytes([0x80|n]) if n<126 else bytes([0x80|126])+struct.pack('!H',n)
 s.sendall(bytes([0x80|opcode])+length+mask+bytes(v^mask[i%4] for i,v in enumerate(payload)))
end=time.monotonic()+args.seconds;count=0;channels={};subs={}
with open(args.output,'w') as out:
 while time.monotonic()<end:
  try:
   a,b=read(2);n=b&127
   if n==126:n=struct.unpack('!H',read(2))[0]
   elif n==127:n=struct.unpack('!Q',read(8))[0]
   mask=read(4) if b&128 else None;data=read(n)
   if mask:data=bytes(v^mask[i%4] for i,v in enumerate(data))
   opcode=a&15
   if opcode==9:send(data,10);continue
   if opcode==8:break
   if opcode==1:
    message=json.loads(data)
    if message.get('op')=='advertise':
     selected=[]
     for c in message['channels']:
      if c['topic']=='/vision/control/state' and c['encoding']=='json':
       subs[c['id']]=c['topic'];selected.append({'id':c['id'],'channelId':c['id']})
     if selected:send(json.dumps({'op':'subscribe','subscriptions':selected}).encode());print('subscribed',selected,flush=True)
   elif opcode==2 and data and data[0]==1:
    sid=struct.unpack_from('<I',data,1)[0]
    if sid in subs:
     value=json.loads(data[13:]);out.write(json.dumps(value)+'\n');count+=1
  except socket.timeout:continue
s.close();print('samples',count)
