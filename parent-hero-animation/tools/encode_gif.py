"""GIF encoder for the email covers.
- one shared 255-colour palette: 185 'scene' colours + 70 'accent' colours (confetti, key, keyhole, sparks)
- accent colours are only allowed where the source pixel is itself accent-like, so they never leak into the shadow or gradient
- delta frames compare against what is actually on screen, so slow fades are not left stale
- identical consecutive frames are merged into one longer frame
usage: python3 encode_final.py out.gif [frame_ms] [box x0,y0,x1,y1] [shadow box]"""
import sys, glob, os, numpy as np, cv2
from PIL import Image
out=sys.argv[1]; DUR=int(sys.argv[2]) if len(sys.argv)>2 else 50
box=tuple(map(int,sys.argv[3].split(','))) if len(sys.argv)>3 else (300,0,600,250)
sbox=tuple(map(int,sys.argv[4].split(','))) if len(sys.argv)>4 else (330,30,500,150)
files=sorted(glob.glob('frames/f*.png'))
frames=[Image.open(f).convert('RGB') for f in files]
keep=[frames[0]]; durs=[DUR]
for f in frames[1:]:
    if (np.abs(np.asarray(f,int)-np.asarray(keep[-1],int)).sum(2)>6).sum()==0: durs[-1]+=DUR
    else: keep.append(f); durs.append(DUR)
frames=keep; w,h=frames[0].size

def accent_mask(arr):
    a=arr.astype(np.float32)/255; mx=a.max(-1); mn=a.min(-1); sat=(mx-mn)/np.maximum(mx,1e-3)
    blue=(a[...,2]>=a[...,0])&(a[...,2]>=a[...,1]-0.02)
    return ((sat>0.3)&~blue)|(mx<0.3)

# ---- palette ----
picks=list(range(0,len(frames),max(1,len(frames)//14)))
fine=list(range(0,len(frames),max(1,len(frames)//40)))
scene_px=[]; acc_px=[]
for p in picks:
    a=np.asarray(frames[p]).reshape(-1,3); m=accent_mask(a); scene_px.append(a[~m][::3])
for p in fine:
    for b,rep in ((box,2),(sbox,4)):
        a=np.asarray(frames[p].crop(b)).reshape(-1,3); m=accent_mask(a)
        for _ in range(rep): scene_px.append(a[~m])
    a=np.asarray(frames[p].crop(box)).reshape(-1,3); acc_px.append(a[accent_mask(a)])
# logo (top centre) and headline get their own weight so their colours are exact
for b,rep in (((40,40,200,85),6),((40,125,260,235),10),((40,235,310,280),4),((300,0,600,380),2)):
    a=np.asarray(frames[0].crop(b)).reshape(-1,3)
    for _ in range(rep): scene_px.append(a)
def mc(px,n):
    px=np.concatenate(px); side=int(np.ceil(np.sqrt(len(px)))); pad=np.zeros((side*side,3),np.uint8); pad[:len(px)]=px; pad[len(px):]=px[0]
    return Image.fromarray(pad.reshape(side,side,3)).quantize(colors=n,method=Image.Quantize.MEDIANCUT).getpalette()[:n*3]
face=[]
for p in fine:
    a_=np.asarray(frames[p].crop(tuple(int(v) for v in os.environ.get("FACE","380,280,500,380").split(",")))).reshape(-1,3)
    for _ in range(3): face.append(a_)
TEXT_BOXES=[(40,40,200,85),(40,125,260,235),(40,235,310,280)]
tp=[np.asarray(frames[0].crop(bx)).reshape(-1,3) for bx in TEXT_BOXES]
allt=np.concatenate(tp); warm=allt[(allt[:,0].astype(int)-allt[:,2].astype(int))>60]   # the orange underline
p_text=mc(tp+[warm]*30,48)+mc([warm],8)                                   # reserved colours for the static logo, headline and underline
px=scene_px+acc_px+face
pal_all=p_text+mc(px,199)
full=Image.new('P',(1,1)); full.putpalette(pal_all+pal_all[:3])
PAL=np.array((pal_all+pal_all[:3])[:256*3]).reshape(-1,3)
textmask=np.zeros((h,w),bool)
for x0,y0,x1,y1 in TEXT_BOXES: textmask[y0:y1,x0:x1]=True
NS=255
THR_K=float(os.environ.get('THR_K','1'))
def quant(f):
    q=np.asarray(f.quantize(palette=full,dither=Image.Dither.FLOYDSTEINBERG)).copy()
    nd=np.asarray(f.quantize(palette=full,dither=Image.Dither.NONE))      # static text: nearest colour, no dither speckle
    q[textmask]=nd[textmask]; return q

# ---- frames: delta against what is on screen ----
out_frames=[]; disp=None; prev_idx=None
for i,f in enumerate(frames):
    q=quant(f); src=np.asarray(f).astype(np.int16)
    if disp is None:
        idx=q
    else:
        thr=8 if i%6==0 else 24                      # every 8th frame catches up on slow fades so nothing is left behind
        B=lambda x: cv2.blur(x.astype(np.float32),(3,3))
        need=((np.abs(B(src)-B(disp)).sum(2)>thr*THR_K)|(np.abs(src-disp).sum(2)>90)).astype(np.uint8)
        need=cv2.dilate(need,np.ones((3,3),np.uint8)).astype(bool)
        idx=np.where(need,q,prev_idx).astype(np.uint8)
    disp=PAL[idx].astype(np.int16)
    arr=idx if prev_idx is None else np.where(idx==prev_idx,255,idx).astype(np.uint8)
    im=Image.fromarray(arr,'P'); im.putpalette(PAL.flatten().tolist()); out_frames.append(im)
    prev_idx=idx
out_frames[0].save(out,save_all=True,append_images=out_frames[1:],duration=durs,loop=0,transparency=255,disposal=1,optimize=False)
print(out,len(out_frames),'frames',w,'x',h,round(os.path.getsize(out)/1024),'KB','total',sum(durs)/1000,'s')
