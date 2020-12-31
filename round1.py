import numpy as np
import math
import cmath
import cv2
import cv2.aruco as aruco
import serial
import time
ser= serial.Serial('COM4',9600)
cap= cv2.VideoCapture(1)

def takepic(cap):
    
    _,frame= cap.read()
    fromCenter=False
    showCrosshair=False
    r= cv2.selectROI('image',frame,fromCenter,showCrosshair)
    img= frame[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
                
    return r,img
def crop(cap,r):
    
    _,frame= cap.read()
    roi= frame[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
        
    return roi

        
        
from collections import deque, namedtuple


# we'll use infinity as a default distance to nodes.
inf = float('inf')
Edge = namedtuple('Edge', 'start, end, cost')


def make_edge(start, end, cost=1):
  return Edge(start, end, cost)


class Graph:
    def __init__(self, edges):
        # let's check that the data is right
        wrong_edges = [i for i in edges if len(i) not in [2, 3]]
        if wrong_edges:
            raise ValueError('Wrong edges data: {}'.format(wrong_edges))

        self.edges = [make_edge(*edge) for edge in edges]

    @property
    def vertices(self):
        return set(
            sum(
                ([edge.start, edge.end] for edge in self.edges), []
            )
        )

    def get_node_pairs(self, n1, n2, both_ends=True):
        if both_ends:
            node_pairs = [[n1, n2], [n2, n1]]
        else:
            node_pairs = [[n1, n2]]
        return node_pairs

    def remove_edge(self, n1, n2, both_ends=True):
        node_pairs = self.get_node_pairs(n1, n2, both_ends)
        edges = self.edges[:]
        for edge in edges:
            if [edge.start, edge.end] in node_pairs:
                self.edges.remove(edge)

    def add_edge(self, n1, n2, cost=1, both_ends=True):
        node_pairs = self.get_node_pairs(n1, n2, both_ends)
        for edge in self.edges:
            if [edge.start, edge.end] in node_pairs:
                return ValueError('Edge {} {} already exists'.format(n1, n2))

        self.edges.append(Edge(start=n1, end=n2, cost=cost))
        if both_ends:
            self.edges.append(Edge(start=n2, end=n1, cost=cost))

    @property
    def neighbours(self):
        neighbours = {vertex: set() for vertex in self.vertices}
        for edge in self.edges:
            neighbours[edge.start].add((edge.end, edge.cost))

        return neighbours

    def dijkstra(self, source, dest):
        assert source in self.vertices, 'Such source node doesn\'t exist'
        distances = {vertex: inf for vertex in self.vertices}
        previous_vertices = {
            vertex: None for vertex in self.vertices
        }
        distances[source] = 0
        vertices = self.vertices.copy()

        while vertices:
            current_vertex = min(
                vertices, key=lambda vertex: distances[vertex])
            vertices.remove(current_vertex)
            if distances[current_vertex] == inf:
                break
            for neighbour, cost in self.neighbours[current_vertex]:
                alternative_route = distances[current_vertex] + cost
                if alternative_route < distances[neighbour]:
                    distances[neighbour] = alternative_route
                    previous_vertices[neighbour] = current_vertex

        path, current_vertex = deque(), dest
        while previous_vertices[current_vertex] is not None:
            path.appendleft(current_vertex)
            current_vertex = previous_vertices[current_vertex]
        if path:
            path.appendleft(current_vertex)
        return path
def graphing(s,g):
    yc1=[]
    ys1=[]
    rc1=[]
    rs1=[]
    gs=[]
    bs=[]
    ws=[]
    k2=[]
    for i in range(9):
      for j in range(9):
        if shape[i][j]==300:
          yc1.append(s[i][j])
        elif shape[i][j]==400:
          ys1.append(s[i][j])
        elif shape[i][j]==3000:
          rc1.append(s[i][j])
        elif shape[i][j]==4000:
          rs1.append(s[i][j])
        elif shape[i][j]==1000:
          gs.append(s[i][j])
        elif shape[i][j]==2000:
          bs.append(s[i][j])
        elif shape[i][j]==700:
          ws.append(s[i][j])
        
    if g==400:
        k2=ys1
    elif g==300:
        k2=yc1
    elif g==3000:
        k2=rc1
    elif g==4000:
        k2=rs1
    else:
        k2=[1000]
    print('k2 list:',k2)
    
    
    
    

    
    
    m=[]
    for i in range(9):
        for j in range(9):
            if i+1<9:
                
                if s[i+1][j] in gs:
                    d1=100
                elif s[i+1][j] in bs:
                    d1=100
                elif s[i+1][j] in k2:
                    d1=2
                elif s[i+1][j] in ws:
                    d1=100
                else:
                    d1=10
    
                m.append((s[i][j],s[i+1][j],d1))
    for i in range(9):
        for j in range(9):
            if j+1<9:
                
                if s[i][j+1] in gs:
                    d2=100
                elif s[i][j+1] in bs:
                    d2=100
                elif s[i][j+1] in k2:
                    d2=2
                elif s[i][j+1] in ws:
                    d2=100
                else:
                    d2=10
                m.append((s[i][j],s[i][j+1],d2))

    for i in range(9):
        for j in range(9):
            if i-1>=0:
                
                if s[i-1][j] in gs:
                    d3=100
                elif s[i-1][j] in bs:
                    d3=100
                elif s[i-1][j] in k2:
                    d3=2
                elif s[i-1][j] in ws:
                    d3=100
                else:
                    d3=10
                m.append((s[i][j],s[i-1][j],d3))

    for i in range(9):
        for j in range(9):
            if j-1>=0:
                
                if s[i][j-1] in gs:
                    d4=100
                elif s[i][j-1] in bs:
                    d4=100
                elif s[i][j-1] in k2:
                    d4=2
                elif s[i][j-1] in ws:
                    d4=100
                else:
                    d4=10
                m.append((s[i][j],s[i][j-1],d4))
            
    return m
                
        

    
def dist(x1,x2,y1,y2):
    a= ((x1-x2)**2+(y1-y2)**2)**(0.5)
    return a

def namebox(roi):
    x=roi[0]
    y=roi[1]
    w=roi[2]
    h=roi[3]
    a=np.zeros(405,dtype=int).reshape(81,5)
    b= np.zeros(81,dtype=int).reshape(9,9)
    p= np.zeros(81,dtype=int).reshape(9,9)
    s= np.zeros(81,dtype=int).reshape(9,9)
    t=1
    for j in range(9):
        for i in range(9):
            b[j][i]= x+(w/18)*(2*i+1)
            p[j][i]= y+(h/18)*(2*j+1)
    for i in range(9):
        for j in range(9):
          s[i][j]=t
          t=t+1
    d=b.reshape(81,1)
    g=p.reshape(81,1)
    print(d)
    print(g)
    print(s)

    for i in range(81):
        a[i][0]=d[i]
        a[i][1]=g[i]
        a[i][2]=i+1
    for i in range(81):
        for m in range(9):
            for n in range(9):
                if a[i][2]==s[m][n]:
                    a[i][3]=m
                    a[i][4]=n
    print(a)
    
    return a,s

def thresh():
    
    m=np.zeros(30,dtype=int).reshape(5,6)
    for i in range(5):
        fromCenter=False
        showCrosshair=False
        r= cv2.selectROI('image',img,fromCenter,showCrosshair)
        imcrop= img[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
        m[i][0]=(imcrop[:,:,0].min())-40
        m[i][1]=(imcrop[:,:,1].min())-40
        m[i][2]=(imcrop[:,:,2].min())-40
        m[i][3]=(imcrop[:,:,0].max())+40
        print(m[i][0])
        print(m[i][3])
        m[i][4]=(imcrop[:,:,1].max())+40
        m[i][5]=(imcrop[:,:,2].max())+40
        lower_red1=np.array([m[i][0],m[i][1],m[i][2]])
        upper_red1=np.array([m[i][3],m[i][4],m[i][5]])
        mask1= cv2.inRange(img,lower_red1, upper_red1)
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(mask1,kernel,iterations = 1)
        dilation = cv2.dilate(erosion,kernel,iterations = 1)
        cv2.imshow('MASK',dilation)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    return m
    
    
def matrix(img,k):    
    def detection(img, i, j,k):

        k1=int(r[2]/9)
        k2=int(r[3]/9)
            
        d=0
        ##--------------------------------------------------green-----------------------------------------##
        lg=np.array([k[1][0],k[1][1],k[1][2]])
        ug=np.array([k[1][3],k[1][4],k[1][5]])
        mask_g= cv2.inRange(img, lg, ug)


        cntsg,_ = cv2.findContours(mask_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnts in cntsg:
            area= cv2.contourArea(cnts)
            if area<2000:
            
                xg, yg, wg, hg =cv2.boundingRect(cnts)
    ##----------------------------------------------------yellow circle and squre inside green--------------------------------------------------##
                hsv= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
                ly=np.array([20,30,50])
                uy= np.array([40,255,255])
                mask_y=cv2.inRange(hsv, ly, uy)
                cntsy,_ = cv2.findContours(mask_y, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for c in cntsy:
                    areat= cv2.contourArea(c)
                    approx= cv2.approxPolyDP(c, 0.03*cv2.arcLength(c,True),True)
                    
                        
                    xy, yy, wy, hy = cv2.boundingRect(c)
                    if (xy>xg and yy>yg and xy+wy<xg+wg and yy+hy<yg+hg):
                        if areat>100:
                            M= cv2.moments(c)
                            cx= int(M["m10"]/M["m00"])
                            cy= int(M["m01"]/M["m00"])
                            cx1=int(cx/k1)
                            cy1=int(cy/k2)
                            if cx1==i and cy1==j:
                                
                                if len(approx)==4:
                                  d=1
                                  return 401
                                else:
                                  d=1
                                  return 301
                            
                ##-------------------------------------------red circle and squre inside green---------------------------##
                lr=np.array([k[2][0],k[2][1],k[2][2]])
                ur=np.array([k[2][3],k[2][4],k[2][5]])
                mask_r=cv2.inRange(img, lr,ur)
                cntsr,_= cv2.findContours(mask_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for c in cntsr:
                    areat= cv2.contourArea(c)
                    approx= cv2.approxPolyDP(c, 0.03*cv2.arcLength(c,True),True)
                    
                        
                    xr, yr, wr, hr = cv2.boundingRect(c)
                    if (xr>xg and yr>yg and xr+wr<xg+wg and yr+hr<yg+hg):
                        if areat>100:
                            M= cv2.moments(c)
                            cx= int(M["m10"]/M["m00"])
                            cy= int(M["m01"]/M["m00"])
                            cx1=int(cx/k1)
                            cy1=int(cy/k2)
                            if cx1==i and cy1==j:
                                
                                if len(approx)==4:
                                  d=1
                                  return 4001
                                else:
                                  d=1
                                  return 3001
                            
                ##-------------------------------white inside green--------------------------##
                lrw=np.array([0,0,190])
                urw= np.array([180,43,255])
                mask_w=cv2.inRange(hsv, lrw, urw)
                cntsw,_= cv2.findContours(mask_w, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for c in cntsw:
                    areat= cv2.contourArea(c)
                    approx= cv2.approxPolyDP(c, 0.03*cv2.arcLength(c,True),True)
                    
                    if areat>20:    
                        xw, yw, ww, hw = cv2.boundingRect(c)
                        if (xw>xg and yw>yg and xw+ww<xg+wg and yw+hw<yg+hg):
                            if areat>100:
                                M= cv2.moments(c)
                                cx= int(M["m10"]/M["m00"])
                                cy= int(M["m01"]/M["m00"])
                                cx1=int(cx/k1)
                                cy1=int(cy/k2)
                                if cx1==i and cy1==j:
                                    
                                    if len(approx)==4:
                                      d=1
                                      return 1000
                                    
                                
        ##-----------------------------------------------only yellow squre and circle----------------------------------##
        hsv= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        ly=np.array([20,30,50])
        uy= np.array([40,255,255])
        mask_y=cv2.inRange(hsv, ly, uy)
        lr=np.array([k[2][0],k[2][1],k[2][2]])
        ur=np.array([k[2][3],k[2][4],k[2][5]])
        mask_r=cv2.inRange(img, lr,ur)
        cntsy,_= cv2.findContours(mask_y, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cntsr,_= cv2.findContours(mask_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in cntsy:
            areay=cv2.contourArea(c)
            approx= cv2.approxPolyDP(c, 0.03*cv2.arcLength(c, True), True)
            if areay>100:
              M= cv2.moments(c)
              cx=int(M["m10"]/M["m00"])
              cy=int(M["m01"]/M["m00"])
              cx1=int(cx/k1)
              cy1=int(cy/k2)
              if cx1==i and cy1==j:
                if len(approx)==4:
                  d=1
                  return 400
                else:
                  d=1
                  return 300
                  
              
        ##--------------------------------------------------------only red circle and squre------------------------------------------------##
        for c in cntsr:
            arear=cv2.contourArea(c)
            approx= cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, True), True)
            if arear>100:
              M= cv2.moments(c)
              cx=int(M["m10"]/M["m00"])
              cy=int(M["m01"]/M["m00"])
              cx1=int(cx/k1)
              cy1=int(cy/k2)
              if cx1==i and cy1==j:
                if len(approx)==4:
                  d=1
                  return 4000
                else:
                  d=1
                  return 3000
              
        ##-----------------------------------------------blue squre-------------------------------------------------##

        ##---------------------------------------------------------only white box-------------------------------------------##
        lrw=np.array([k[3][0],k[3][1],k[3][2]])
        urw=np.array([k[3][3],k[3][4],k[3][5]])
        mask_w= cv2.inRange(img, lrw, urw)
        cntsw,_= cv2.findContours(mask_w, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in cntsw:
            areaw=cv2.contourArea(c)
            approx= cv2.approxPolyDP(c, 0.03*cv2.arcLength(c, True), True)
            if areaw>100:
              M= cv2.moments(c)
              cx=int(M["m10"]/M["m00"])
              cy=int(M["m01"]/M["m00"])
              cx1=int(cx/k1)
              cy1=int(cy/k2)
              if cx1==i and cy1==j:
                if len(approx)==4:
                  d=1
                  return 700
                
                  
              
        ##------------------------------------------------------none of these-----------------------------------------------##
        if d==0:
            return 2000
          
  
    shape=np.zeros(81,dtype=int).reshape(9,9)
    mm=[41,76,77,78]
    for i in range(9):
      for j in range(9):
          if s[j][i]in mm:
              
              shape[j][i]=2000
          else:
              shape[j][i]=detection(img,i,j,k)
              
    return shape

def arucos():
    while (True):
        _, img = cap.read()
        
        gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        aruco_dict= aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters= aruco.DetectorParameters_create()
        corners, ids,_=aruco.detectMarkers(gray, aruco_dict, parameters= parameters)
        print(ids,corners)
        if(len(corners)!=0):
            break
    
    A1=[]
    A2=[]

    for i in range(4):
        A1.append(corners[0][0][i][0])
        A2.append(corners[0][0][i][1])
    return A1,A2

def go1(cx,cy,cap,r):
                
    A1, A2= arucos()
    
    A3=np.array(A1)
    A4=np.array(A2)
    nx=A3.sum()/4
    ny=A4.sum()/4
    d=dist(nx,cx,ny,cy)
    z1=complex(cx-nx,cy-ny)
    z2=complex(A1[1]-A1[2],A2[1]-A2[2])
    print(z1,z2)

    c3=np.angle(z1,deg=True)
    c4=np.angle(z2,deg=True)
    print(c3)
    print(c4)

    h=0-c4
    c2=c3+h
    if c2>180:
        c1=c2-360
    elif c2<-180:
        c1=c2+360
    else:
        c1=c2
    print(c1)

    if c1>-15 and c1<15:
        if d<20:
          return 's'
        
        else:
          return 'f'
    else:
        if c1 > 15 and c1<=180:
          return 'r'
        elif c1 < -15 or c1> -180:
          return 'l'
def align(cx,cy,cap,r):
    while(True):
        _,img= cap.read()
        break

    
                
                
    A1, A2= arucos()
    
    A3=np.array(A1)
    A4=np.array(A2)
    nx=A3.sum()/4
    ny=A4.sum()/4
    d=dist(nx,cx,ny,cy)
    z1=complex(cx-nx,cy-ny)
    z2=complex(A1[1]-A1[2],A2[1]-A2[2])
    print(z1,z2)
    c3=np.angle(z1,deg=True)
    c4=np.angle(z2,deg=True)
    print(c3)
    print(c4)

    h=0-c4
    c2=c3+h
    if c2>180:
        c1=c2-360
    elif c2<-180:
        c1=c2+360
    else:
        c1=c2
    print(c1)
    
    if c1>-15 and c1<15:
        if d<50:
          return 's'
        
        else:
          return 'f'
        
        
    else:
        if c1 > 15 and c1<=180:
          return 'r'
        elif c1 < -15 or c1> -180:
          return 'l'
##---------------------------------algorithm--------------------------------------##
n1=9
n2=81

r,img= takepic(cap)
a,s=namebox(r)
k=thresh()
shape=matrix(img,k)
print(shape)
m=graphing(s,0)
graph=Graph(m)
b=[]
g=[]
w=[]
for i in range(n1):
    for j in range(n1):
        if shape[i][j]==2000:
            b.append(s[i][j])
        if shape[i][j]==700:
            w.append(s[i][j])
        if shape[i][j]==1000:
            g.append(s[i][j])
print(b)
print(w)
print(g)                    
j1,j2=arucos()
j3=np.array(j1)
j4=np.array(j2)
jx=j3.sum()/4
jy=j4.sum()/4
p1=r[2]/5
p2=r[3]/5
jx1=jx-r[0]
jy1=jy-r[1]
print(p1)
print(p2)
i2=int(jx1/p1)
i1=int(jy1/p2)
print(i1)
print(i2)
source=s[i1][i2]
print(source)

time.sleep(2)

for i in range(len(g)):
    
    dest=g[i]
    
    P=graph.dijkstra(source,dest)
    print(P)
    time.sleep(5)

    t=0
    while(True):
        if t==0:
            t=1
            
        for c in range(n2):
            if P[t]==a[c][2]:
                cx=a[c][0]

                cy=a[c][1]
            
        print('GOING to:',P[t])
        if go1(cx,cy,cap,r)=='f':
            ser.write(b'F')
            print('F')
            time.sleep(0.2)
            ser.write(b'S')
            print('S')

        elif go1(cx,cy,cap,r)=='r':
            ser.write(b'R')
            print('R')
            time.sleep(0.1)
            ser.write(b'S')
            print('S')
    
        elif go1(cx,cy,cap,r)=='l':
            ser.write(b'L')
            print('L')
            time.sleep(0.1)
            ser.write(b'S')
            print('S')
    
        elif go1(cx,cy,cap,r)=='s':
            ser.write(b'S')
            print('S')
            time.sleep(0.1)
            print('-----------------T---------------------:',t)
            t=t+1
            if t == len(P)-1:
                break
            
    for c in range(n2):
            if P[t]==a[c][2]:
                cx=a[c][0]

                cy=a[c][1]
    while(1):
        if align(cx,cy,cap,r)=='r':
            ser.write(b'R')
            print('R')
            time.sleep(0.1)
            ser.write(b'S')
            print('S')
        elif align(cx,cy,cap,r)=='l':
            ser.write(b'L')
            print('L')
            time.sleep(0.1)
            ser.write(b'S')
            print('S')
        elif align(cx,cy,cap,r)=='f':
            ser.write(b'F')
            print('F')
            time.sleep(0.2)
            ser.write(b'S')
            print('S')
        elif align(cx,cy,cap,r)=='s':
            ser.write(b'S')
            time.sleep(0.2)
            break
    ser.write(b'D')
    print('Servo Down')
    source=P[len(P)-2]
    dest=b[i]

    Q=graph.dijkstra(source,dest)
    print(Q)
    x=0
    while(True):
        if x==0:
            x=1
            
        for c in range(n2):
            if Q[x]==a[c][2]:
                cx=a[c][0]

                cy=a[c][1]
            
        print('GOING to:',Q[x])
        if go1(cx,cy,cap,r)=='f':
            ser.write(b'F')
            print('F')
            time.sleep(0.2)
            ser.write(b'S')
            print('S')
        elif go1(cx,cy,cap,r)=='r':
            ser.write(b'R')
            print('R')
            time.sleep(0.1)
            ser.write(b'S')
            print('S')
        elif go1(cx,cy,cap,r)=='l':
            ser.write(b'L')
            print('L')
            time.sleep(0.1)
            ser.write(b'S')
            print('S')
        elif go1(cx,cy,cap,r)=='s':
            ser.write(b'S')
            print('S')
            time.sleep(0.2)
            x=x+1
        if x==len(Q)-1:
            break
    for c in range(n2):
            if Q[x]==a[c][2]:
                cx=a[c][0]

                cy=a[c][1]
    while(1):
        if align(cx,cy,cap,r)=='r':
            ser.write(b'R')
            print('R')
            time.sleep(0.1)
            ser.write(b'S')
            print('S')
        elif align(cx,cy,cap,r)=='l':
            ser.write(b'L')
            print('L')
            time.sleep(0.1)
            ser.write(b'S')
            print('S')
        elif align(cx,cy,cap,r)=='f':
            ser.write(b'F')
            print('F')
            time.sleep(0.2)
            ser.write(b'S')
            print('S')
        elif align(cx,cy,cap,r)=='s':
            ser.write(b'S')
            time.sleep(0.2)
            break
    ser.write(b'U')
    print('Servo Up')
    ser.write(b'B')
    
    source=Q[len(Q)-2]


for h in range(len(w)):
    
    dest=w[h]
    for i in range(n1):
        for j in range(n1):
            if w[h]==s[i][j]:
                b1=i
                b2=j
        
           

    G=graph.dijkstra(source,dest)
    time.sleep(3)
    
    t1=0
    while(True):
        if t1==0:
            t1=1
            
        for c in range(n2):
            if G[t1]==a[c][2]:
                cx=a[c][0]

                cy=a[c][1]
            
        print('GOING to:',G[t1])
        if go1(cx,cy,cap,r)=='f':
            ser.write(b'F')
            print('F')
            time.sleep(0.2)
            ser.write(b'S')
            print('S')
        elif go1(cx,cy,cap,r)=='r':
            ser.write(b'R')
            print('R')
            time.sleep(0.1)
            ser.write(b'S')
            print('S')
        elif go1(cx,cy,cap,r)=='l':
            ser.write(b'L')
            print('L')
            time.sleep(0.1)
            ser.write(b'S')
            print('S')
        elif go1(cx,cy,cap,r)=='s':
            ser.write(b'S')
            print('S')
            time.sleep(0.1)
            t1=t1+1
            if t1 == len(G)-1:
                break
            
    for c in range(n2):
            if G[t1]==a[c][2]:
                cx=a[c][0]

                cy=a[c][1]
    while(1):
        if align(cx,cy,cap,r)=='r':
            ser.write(b'R')
            print('R')
            time.sleep(0.1)
            ser.write(b'S')
            print('S')
        elif align(cx,cy,cap,r)=='l':
            ser.write(b'L')
            print('L')
            time.sleep(0.1)
            ser.write(b'S')
            print('S')
        elif align(cx,cy,cap,r)=='f':
            ser.write(b'F')
            print('F')
            time.sleep(0.2)
            ser.write(b'S')
            print('S')
        elif align(cx,cy,cap,r)=='s':
            ser.write(b'S')
            time.sleep(0.2)
            break
    ser.write(b'D')
    ser.write(b'G')
    print('--------------------Servo Down-----------------------')
    time.sleep(2)

    for c in range(n2):        
        if G[t1-2]==a[c][2] :
            cx=a[c][0]
            cy=a[c][1]
        
    while(1):
        if align(cx,cy,cap,r)=='r':
            ser.write(b'R')
            print('R')
            time.sleep(0.1)
            ser.write(b'S')
            print('S')
        elif align(cx,cy,cap,r)=='l':
            ser.write(b'L')
            print('L')
            time.sleep(0.1)
            ser.write(b'S')
            print('S')
        elif align(cx,cy,cap,r)=='f':
            ser.write(b'F')
            print('F')
            time.sleep(0.2)
            ser.write(b'S')
            print('S')
        elif align(cx,cy,cap,r)=='s':
            ser.write(b'S')
            time.sleep(3)
            frame1= crop(cap,r)
            shape1= matrix(frame1,k)
            print(shape1)
            print(shape1[b1][b2])
            if shape1[b1][b2]==300:
                shape[b1][b2]=300
            elif shape1[b1][b2]==3000:
                shape[b1][b2]=3000
            elif shape1[b1][b2]==400:
                shape[b1][b2]=400
            elif shape1[b1][b2]==4000:
                shape[b1][b2]=4000
            break


    print('UNder the box:',b1,b2,shape[b1][b2])
    time.sleep(2)

    d=graphing(s,shape[b1][b2])
    graph=Graph(d)
    source=G[t1-1]
    for i in range(n1):
        for j in range(n1):
            if shape[i][j]==shape[b1][b2]+1:
                dest=s[i][j]
    G1=graph.dijkstra(source,dest)
    print(G1)
    time.sleep(3)

    x1=0
    while(True):
        
        if x1==0:
            x1=1
            
        for c in range(n2):
            if G1[x1]==a[c][2]:
                cx=a[c][0]

                cy=a[c][1]
            
        print('GOING to:',G1[x1])
        if go1(cx,cy,cap,r)=='f':
            ser.write(b'F')
            print('F')
            time.sleep(0.2)
            ser.write(b'S')
            print('S')
        elif go1(cx,cy,cap,r)=='r':
            ser.write(b'R')
            print('R')
            time.sleep(0.1)
            ser.write(b'S')
            print('S')
        elif go1(cx,cy,cap,r)=='l':
            ser.write(b'L')
            print('L')
            time.sleep(0.1)
            ser.write(b'S')
            print('S')
        elif go1(cx,cy,cap,r)=='s':
            ser.write(b'S')
            print('S')
            time.sleep(0.2)
            x1=x1+1
        if x1==len(G1)-1:
            break
    for c in range(n2):
            if G1[x1]==a[c][2]:
                cx=a[c][0]

                cy=a[c][1]
    while(1):
        if align(cx,cy,cap,r)=='r':
            ser.write(b'R')
            print('R')
            time.sleep(0.1)
            ser.write(b'S')
            print('S')
        elif align(cx,cy,cap,r)=='l':
            ser.write(b'L')
            print('L')
            time.sleep(0.1)
            ser.write(b'S')
            print('S')
        elif align(cx,cy,cap,r)=='f':
            ser.write(b'F')
            print('F')
            time.sleep(0.2)
            ser.write(b'S')
            print('S')
        elif align(cx,cy,cap,r)=='s':
            ser.write(b'S')
            time.sleep(0.2)
            break
    ser.write(b'U')
    ser.write(b'H')
    print('Servo Up')
    
    source=G1[len(G1)-2]
print('Finished')
ser.write(b'T')


print('Finished')
