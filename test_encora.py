# Problem :  

# You have to implement a point animation screen, something similar to this: https://www.youtube.com/watch?v=5mGuCdlCcNM

 
# For this problem 

# - Assume you already have a grid to draw(x, y) or erase(x, y) 

# - Assume you already have functions, whose will draw and erase 

# - The point/ball needs to bounce in each edge of the screen 
x_dvd, y_dvd = 100, 100

def draw(x, y):
    pass

def erase(x, y):
    pass

def movimiento(x_dvd,y_dvd):
    draw(x_dvd, y_dvd)
    time.sleep(2)
    erase(x_dvd, y_dvd)

def validacion():
    if x_dvd+5 <= x_limit and y_dvd+5 <= y_limit:
        movimiento(x_dvd+5, y_dvd+5)
        x_dvd = x_dvd+5
        y_dvd = y_dvd+5
        
    elif x_dvd+5 > x_limit:
        movimiento(x_dvd-5, y_dvd)
        x_dvd = x_dvd+5
        y_dvd = y_dvd-5
        
    elif y_dvd+5 > y_limit:
        movimiento(x_dvd+5, y_dvd-5)
        x_dvd = x_dvd+5
        y_dvd = y_dvd-5
    
x_limit = 200
y_limit = 200

while:
    validacion()

  A  
########################    
#          #
#          #
A#         #
#          #
#          #100 *
#          #  *
#          #*
#         *#
#    *  *  #
#     *    #
############

200, 100

 B
############    
#          #
#          #
#          #
#          #
#          #
#          #
#          #
#         *#
#    *  *  #
#     *    #
############
