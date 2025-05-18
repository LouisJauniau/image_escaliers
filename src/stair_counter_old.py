import cv2
import numpy as np
import math

def count_stairs_approachB_visual( #Configuration
    image_path,
    blur_ksize=(7,7),
    canny_thresh1=80,
    canny_thresh2=200,
    close_kernel_size=(3,3),
    hough_threshold=50,
    min_line_length=100,
    max_line_gap=10,
    angle_tolerance=5, 
    y_group_distance=15, 
    discard_small_groups=True,
    group_min_size=2,
    **_ignored  
):

    img_color = cv2.imread(image_path) #Chargement de l'image

    if img_color is None: #Verifie si l'image est chargée
        print(f"Impossible de lire l'image : {image_path}")
        return 0
    
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY) #Conversion en niveaux de gris

    blurred = cv2.GaussianBlur(img_gray, blur_ksize, 0) #Flou gaussien
    edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2) #Detection de contours via Canny
    # canny_thresh1, canny_thresh2 les deux seuils de l'algorithme

    kernel_close = np.ones(close_kernel_size, np.uint8) #Creation du noyau / Operation morphologique 
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close) #Fermeture morphologique / Combler les trous

    lines = cv2.HoughLinesP( #Detection de segments de lignes 
        edges_closed,
        rho=1,
        theta=np.pi/180,
        threshold=hough_threshold, # : nombre minimal de points alignés pour valider une ligne
        minLineLength=30, # : longueur minimale pour qu’un segment soit considéré
        maxLineGap=max_line_gap # : écart maximal entre deux segments
    )
    if lines is None: # Verification si aucunes lignes detectees
        print("Aucune ligne détectée, 0 marche")
        return 0

    horizontal_boxes = [] #Stocke les bounding boxes
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        # Calcul de l'inclinaison
        if dx == 0: # Evite les lignes verticales
            continue
        angle = abs(math.degrees(math.atan2(dy, dx))) # Pour calculer l'angle de la ligne
        if angle <= angle_tolerance: # Compare avec avec l'angle de tolerance
            length = np.hypot(dx, dy) # Verifie la longueur de la ligne
            if length >= min_line_length: 
                bx = min(x1, x2)
                by = min(y1, y2)
                bw = abs(x2 - x1)
                bh = abs(y2 - y1)
                horizontal_boxes.append((bx, by, bw, bh))
                # Alors on calcul les coordonnées de la bounding box

    if not horizontal_boxes: # Verifie si vide
        print("Aucune ligne horizontale détectée, 0 marche")
        return 0

    # Determine la ROI
    min_x = min(bx for (bx, by, bw, bh) in horizontal_boxes)
    min_y = min(by for (bx, by, bw, bh) in horizontal_boxes)
    max_x = max(bx + bw for (bx, by, bw, bh) in horizontal_boxes)
    max_y = max(by + bh for (bx, by, bw, bh) in horizontal_boxes)
    # On trouve les valeurs minimum et maximum de x et y parmi toutes les bounding boxes horizontales détectées
    # min_x, min_y : coin supérieur gauche
    # max_x, max_y : coin inférieur droit

    if min_x >= max_x or min_y >= max_y: # ROI logique ?
        print("Impossible de définir un ROI, 0 marche")
        return 0

    roi_w = max_x - min_x # largeur
    roi_h = max_y - min_y # hauteur

    roi = edges_closed[min_y : min_y + roi_h, min_x : min_x + roi_w] # Extraction de la ROI dans notre image

    lines_roi = cv2.HoughLinesP( # Nouvelle detection dans la ROI
        roi,
        rho=1,
        theta=np.pi/180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    if lines_roi is None:
        print("Aucune ligne détectée dans la ROI, 0 marche")
        return 0

    horizontal_lines = [] # Lignes détectées dans la nouvelle ROI
    for line in lines_roi:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle = abs(math.degrees(math.atan2(dy, dx)))
        if angle <= angle_tolerance:
            horizontal_lines.append((x1, y1, x2, y2))

    if not horizontal_lines: # Verif
        print("Aucune ligne horizontale détectée dans la ROI, 0 marche")
        return 0

    y_means = []
    for (lx1, ly1, lx2, ly2) in horizontal_lines:
        ym = (ly1 + ly2) / 2.0
        y_means.append(ym)
    y_means.sort()

    groups = []
    current_group = [y_means[0]]
    for val in y_means[1:]: # Regroupe les lignes proches
        if abs(val - current_group[-1]) <= y_group_distance:
            current_group.append(val)
        else:
            groups.append(current_group)
            current_group = [val]
    groups.append(current_group)

    if discard_small_groups:
        groups = [g for g in groups if len(g) >= group_min_size]

    nb_marches = len(groups)


    cv2.rectangle(
        img_color,
        (min_x, min_y),
        (max_x, max_y),
        (0, 255, 0),
        2
    )

    roi_color = img_color[min_y : min_y + roi_h, min_x : min_x + roi_w].copy()

    for (lx1, ly1, lx2, ly2) in horizontal_lines:
        cv2.line(roi_color, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)

    img_color[min_y : min_y + roi_h, min_x : min_x + roi_w] = roi_color

    cv2.imshow("Image couleur avec ROI & lignes", img_color)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Nombre de marches détectées : {nb_marches}")
    return nb_marches


if __name__ == "__main__":
    image_path = "82.jpg"
    nb_stairs = count_stairs_approachB_visual(image_path)
    print(f"Approche B, Nombre de marches détectées : {nb_stairs}")
