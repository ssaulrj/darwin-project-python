# HORIZONTAL --------------------------------------------------------------------------------------
        horizontal = np.copy(bw)

        # Specify size on horizontal axis
        cols = horizontal.shape[1]
        horizontal_size = cols / 40
        horizontal_size=int(horizontal_size)
        #print("horizontalsize", horizontal_size)

        # Create structure element for extracting horizontal lines through morphology operations
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        #cv2.imshow("horizontalStructure", horizontalStructure)
        # Apply morphology operations
        horizontal = cv2.erode(horizontal, horizontalStructure)
        horizontal = cv2.dilate(horizontal, horizontalStructure) 
        horizontal = cv2.bitwise_not(horizontal)
        # Step 1
        edges = cv2.adaptiveThreshold(horizontal, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
        # Step 2
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel)
        # Step 3
        smooth = np.copy(horizontal)
        # Step 4
        smooth = cv2.blur(smooth, (2, 2))
        # Step 5
        (rows, cols) = np.where(edges != 0) #(edges != 0)
        horizontal[rows, cols] = smooth[rows, cols]

        # Show extracted horizontal lines
        #if horizontal is not None:
            #cv2.imshow('image_horizontal',horizontal)
# VERTICAL ------------------------------------------------------------------------------------------
        # Create the images that will use to extract the vertical lines
        vertical = np.copy(bw)
        # Specify size on vertical axis
        rows = vertical.shape[0]
        verticalsize = rows / 40 #20
        verticalsize = int(verticalsize)
        #print("verticalsize", verticalsize)

        # Create structure element for extracting vertical lines through morphology operations
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
        #cv2.imshow("verticalStructure", verticalStructure)
        # Apply morphology operations
        vertical = cv2.erode(vertical, verticalStructure)
        vertical = cv2.dilate(vertical, verticalStructure)
        # Inverse vertical image
        vertical = cv2.bitwise_not(vertical) #(vertical)
        # Step 1
        edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
        # Step 2
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel)
        # Step 3
        smooth = np.copy(vertical)
        # Step 4
        smooth = cv2.blur(smooth, (2, 2))
        # Step 5
        (rows, cols) = np.where(edges != 0) #(edges != 0)
        vertical[rows, cols] = smooth[rows, cols]

        #diff_total = cv2.absdiff(vertical, horizontal) # the original image - the difference 
        #cv2.imshow("diff_total", diff_total)
        #edged = cv2.Canny(diff_total, 30, 200)
        #cv2.imshow("edged", edged)
        #contours, hierachy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        #for cnt in contours:
        #    area = cv2.contourArea(cnt)
        #    if area > 100:
        #        cv2.drawContours(diff_total, contours, -1, (0, 255, 0), 3)
        #cv2.imshow("contours", diff_total)

        # Show final result
        #if vertical is not None:
            #cv2.imshow('image_vertical', vertical)

    # Parallel lines ----------------------
        """lines1 = []
        for i in range(len(lines)):
            for j in range(len(lines)):
                if (i == j):continue
                if (abs(lines[i][0][1] - lines[j][0][1]) == 0 and abs(lines[i][0][1]) < 1):          
                    print("You've found a parallel line!", abs(lines[i][0][1]))
                    lines1.append((i,j))
                    rho = lines[i][0][0]
                    theta = lines[i][0][1]
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 10000*(-b))
                    y1 = int(y0 + 10000*(a))
                    x2 = int(x0 - 10000*(-b))
                    y2 = int(y0 - 10000*(a))
                    cv2.line(cdst, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow("lineas2", cdst)"""

                        # Line ends filter
    def lineEnds(P):
        """Central pixel and just one other must be set to be a line end"""
        return 255 * ((P[4]==255) and np.sum(P)==510)

    def paint_region_with_avg_intensity(self, img, rp, mi, channel): #Marcar regiones de segmentacion
        for i in range(rp.shape[0]):
            img[rp[i][0]][rp[i][1]][channel] = mi
        return img

    def seg_superpix(self, img): #Felzenszwalb es mas estable, realizar pruebas con los 3 (slic, watershed & fel..)
        #segments = slic(img, n_segments=200, compactness=10, multichannel=True, enforce_connectivity=True, convert2lab=True)
        segments = felzenszwalb(img, scale=100, sigma=0.7, min_size=10)
        #gradient = sobel(rgb2gray(img))
        #segments = watershed(gradient, markers=250, compactness=0.001)
        for i in range(3):
            regions = regionprops(segments, intensity_image=img[:,:,i])
            for r in regions:
                img = self.paint_region_with_avg_intensity(img, r.coords, int(r.mean_intensity), i)
        return img 

    self.seg_image = self.seg_superpix(self.bg_removed_green_blue)  
                    cv2.imshow("seg_image", self.seg_image)