from __future__ import annotations

import argparse
import math
import pathlib
import sys
from typing import List, Tuple, Union

import cv2
import numpy as np


#Paramètres optimaux


OPTIMAL_PARAMS = {
    "blur_ksize": (7, 7),
    "canny_thresh1": 45,
    "canny_thresh2": 90,
    "close_kernel_size": (3, 3),
    "hough_threshold": 40,
    "min_line_length": 70,
    "max_line_gap": 15,
    "angle_tolerance": 8,
    "y_group_distance": 10,
    "discard_small_groups": True,
    "group_min_size": 2,
    "apply_clahe": False,
    "length_ratio_threshold": 0.3,
}


#Détection du nombre de marches

def count_stairs_approachB_visual(
    image_path: str,
    *,
    return_image: bool = False,
    debug: bool = False,
    **override_params,
) -> Union[int, Tuple[int, np.ndarray]]:


    #Construction des hyper‑paramètres effectifs
    params = OPTIMAL_PARAMS | override_params 

    #Lecture de l'image couleur
    img_color = cv2.imread(image_path)
    if img_color is None:
        raise FileNotFoundError(f"Impossible de lire l'image : {image_path}")

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    if params["apply_clahe"]:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)

    #flou + Canny
    blurred = cv2.GaussianBlur(img_gray, params["blur_ksize"], 0)
    edges = cv2.Canny(blurred, params["canny_thresh1"], params["canny_thresh2"])

    #fermeture morphologique
    kernel_close = np.ones(params["close_kernel_size"], np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)

    #Hough global
    lines = cv2.HoughLinesP(
        edges_closed,
        rho=1,
        theta=np.pi / 180,
        threshold=params["hough_threshold"],
        minLineLength=params["min_line_length"],
        maxLineGap=params["max_line_gap"],
    )
    if lines is None:
        if debug:
            print("aucune ligne globale")
        return (0, img_color) if return_image else 0

    #filtrage des lignes quasi‑horizontales
    horizontal_boxes: List[Tuple[int, int, int, int]] = []  #(x,y,w,h)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:
            continue
        angle = abs(math.degrees(math.atan2(dy, dx)))
        if angle <= params["angle_tolerance"] and math.hypot(dx, dy) >= params["min_line_length"]:
            bx, by = min(x1, x2), min(y1, y2)
            bw, bh = abs(dx), abs(dy)
            horizontal_boxes.append((bx, by, bw, bh))

    if not horizontal_boxes:
        return (0, img_color) if return_image else 0

    #ROI englobant toutes les boîtes horizontales
    min_x = min(bx for bx, by, bw, bh in horizontal_boxes)
    min_y = min(by for bx, by, bw, bh in horizontal_boxes)
    max_x = max(bx + bw for bx, by, bw, bh in horizontal_boxes)
    max_y = max(by + bh for bx, by, bw, bh in horizontal_boxes)

    if min_x >= max_x or min_y >= max_y:
        return (0, img_color) if return_image else 0

    roi = edges_closed[min_y:max_y, min_x:max_x]

    #Hough dans le ROI
    lines_roi = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 180,
        threshold=params["hough_threshold"],
        minLineLength=params["min_line_length"],
        maxLineGap=params["max_line_gap"],
    )
    if lines_roi is None:
        return (0, img_color) if return_image else 0

    #filtrage horizontaux + longueur relative
    horizontal_lines: List[Tuple[int, int, int, int]] = []
    for line in lines_roi:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:
            continue
        angle = abs(math.degrees(math.atan2(dy, dx)))
        if angle <= params["angle_tolerance"]:
            horizontal_lines.append((x1, y1, x2, y2))

    if not horizontal_lines:
        return (0, img_color) if return_image else 0

    max_len = max(abs(x2 - x1) for x1, y1, x2, y2 in horizontal_lines)
    effective_lines = [
        (x1, y1, x2, y2)
        for (x1, y1, x2, y2) in horizontal_lines
        if abs(x2 - x1) >= params["length_ratio_threshold"] * max_len
    ]
    if not effective_lines:
        return (0, img_color) if return_image else 0

    #clustering par position verticale
    y_means = sorted([(ly1 + ly2) / 2.0 for lx1, ly1, lx2, ly2 in effective_lines])
    groups = []
    current = [y_means[0]]
    for y in y_means[1:]:
        if abs(y - current[-1]) <= params["y_group_distance"]:
            current.append(y)
        else:
            groups.append(current)
            current = [y]
    groups.append(current)

    if params["discard_small_groups"]:
        groups = [g for g in groups if len(g) >= params["group_min_size"]]

    nb_marches = len(groups)

    #VISUALISATION
    if return_image or debug:
        vis = img_color.copy()
        # ROI (vert)
        cv2.rectangle(vis, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        #Lignes rouges
        for x1, y1, x2, y2 in effective_lines:
            cv2.line(
                vis,
                (min_x + x1, min_y + y1),
                (min_x + x2, min_y + y2),
                (0, 0, 255),
                2,
            )
    else:
        vis = None

    if debug:
        print(f"[DEBUG] {nb_marches} marches détectées dans {image_path}")

    return (nb_marches, vis) if return_image else nb_marches



def _default_out_path(img_path: pathlib.Path) -> pathlib.Path:
    return img_path.with_suffix("").with_name(img_path.stem + "_annot.jpg")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compter le nombre de marches et enregistrer l'image annotée",
    )
    parser.add_argument("image", help="Chemin vers l'image .jpg/.png")
    parser.add_argument("-o", "--out", help="Fichier de sortie (annoté)")
    parser.add_argument("--show", action="store_true", help="Afficher la fenêtre OpenCV")
    parser.add_argument("--debug", action="store_true", help="Debug verbose")
    args = parser.parse_args()

    img_path = pathlib.Path(args.image)
    if not img_path.exists():
        sys.exit(f"Fichier introuvable : {img_path}")

    #Exécution
    n, vis = count_stairs_approachB_visual(str(img_path), return_image=True, debug=args.debug)
    print(f"Nombre de marches détectées : {n}")

    #Enregistrement / affichage
    out_path = pathlib.Path(args.out) if args.out else _default_out_path(img_path)
    if vis is not None:
        cv2.imwrite(str(out_path), vis)
        print(f"Image annotée enregistrée → {out_path}")
        if args.show:
            cv2.imshow("Stair detection", vis)
            cv2.waitKey(0)


if __name__ == "__main__":
    main()
