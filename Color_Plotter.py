import socket
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from matplotlib.font_manager import FontProperties
import cv2 
import numpy as np
import time
import socket

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)


xs = []
ys = []
gs =[]
rs=[]
hs=[]
ss=[]
vs=[]

cap = cv2.VideoCapture(0)
height=405
width=720
square_size_px=320
area_take_samples_px=50
cap.set(4,height)
cap.set(3,width)
start_time=round (time.time(),4)
def animate(i, xs, ys, gs, rs, hs, ss, vs):
    _,frame=cap.read()
    y=int ((height/2)-(square_size_px/2))
    x=int ((width/2)-(square_size_px/2))
    frame_gets = frame[y:y+square_size_px, x:x+square_size_px]
    frame_gets_cp = frame_gets.copy()
    x_samples=int ((square_size_px/2)-(area_take_samples_px/2))
    y_samples=x_samples
    h_samples=int ((square_size_px/2)+(area_take_samples_px/2))
    w_samples=h_samples
    cv2.rectangle(frame_gets_cp,(x_samples,y_samples),(w_samples,h_samples),(255,0,0),3)
    samples_frame = frame_gets [y_samples:y_samples+area_take_samples_px, x_samples:x_samples+area_take_samples_px]
    hsv_frame= cv2.cvtColor(samples_frame, cv2.COLOR_BGR2HSV)
    blue, green, red = cv2.split(samples_frame)
    h, s, v = cv2.split(hsv_frame)
    b_av= np.average(blue)
    g_av= np.average(green)
    r_av= np.average(red)
    h_av= np.average(h)
    s_av= np.average(s)
    v_av= np.average(v)
    now=time.time()
    time_step = now - start_time

    cv2.imshow('viewer',frame_gets_cp)
    cv2.waitKey(1) 
           
 
    
    # Add x and y to lists
    xs.append(time_step)
    ys.append(b_av)
    rs.append(r_av)
    gs.append(g_av)
    hs.append(h_av)
    ss.append(s_av)
    vs.append(v_av)


    # Limit x and y lists to 20 items
    xs = xs[-20:]
    ys = ys[-20:]
    rs = rs[-20:]
    gs = gs[-20:]
    hs = hs[-20:]
    vs = vs[-20:]
    ss = ss[-20:]

        # Draw x and y lists
    ax.clear()
    ax.plot(xs, ys, color='blue', label='Blue')
    ax.plot(xs, rs, color='red', label='Red')
    ax.plot(xs, gs, color='green', label='Green')
    ax.plot(xs, hs, color='cyan', label='Hue')
    ax.plot(xs, vs, color='magenta',label='Saturation' )
    ax.plot(xs, ss, color='yellow',label='Value')

        # Format plot
    plt.ylim(0, 255)
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('Color Plotter Riset Urine 2021')
    plt.ylabel('Color (Bit)')
    plt.xlabel('Time Step (ms)')
    plt.legend(title='Parameter where:', bbox_to_anchor=(1, 1), loc='upper left')
    
   
if __name__ == '__main__':
    try:
        ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys, rs, gs, hs, vs, ss), interval=1)
        plt.show()
    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()
