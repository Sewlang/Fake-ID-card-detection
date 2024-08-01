from flask import Flask, request, url_for, render_template
import imutils.contours
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity   #Pour comparer la similarité structurelle entre deux images
import cv2 #Pour du image processing
import imutils
from PIL import Image
import os

app = Flask(__name__)
app.config['DOSSIER_UPLOAD'] = "static/uploadedFolder"

@app.route("/")
def fonctionIndex():
    return render_template("index.html")

@app.route("/login")
def fonctionConnecter():
    return render_template("login.html")

@app.route("/validation", methods = ["POST"])
def AccesValide():
    donneeUtilisateur = request.form
    nom = donneeUtilisateur.get('username')
    mdp = donneeUtilisateur.get('password')
    print(type(mdp))
    if nom =="azerty" and mdp == "1234":
        return render_template("UserValide.html",nomutilisateur = nom)
    else:
        return render_template("UserValide.html", nomutilisateur = nom) 

@app.route("/authenticator")
def AuthenticationChecker():
    return render_template("authenticator.html")

@app.route("/traitement", methods = ["POST"])
def traitement():
    if 'image1' not in request.files or 'image2' not in request.files:
        return 'No file part'

    image1 = request.files['image1'] #Image de référence
    image2 = request.files['image2'] #Image a comparer
    chemin_dacces_image1 = (os.path.join(app.config['DOSSIER_UPLOAD'], image1.filename))
    chemin_dacces_image2 = (os.path.join(app.config['DOSSIER_UPLOAD'], image2.filename))

    image1.save(chemin_dacces_image1)
    image2.save(chemin_dacces_image2)
    
    im_original = Image.open(chemin_dacces_image1)
    im_compare = Image.open(chemin_dacces_image2)
    
    largeur_min = min(im_original.size[0],im_compare.size[0])
    hauteur_min = min(im_original.size[1],im_compare.size[1])

    im_original = im_original.resize((largeur_min,hauteur_min))
    im_compare = im_compare.resize((largeur_min,hauteur_min))
    
    im_original.save(chemin_dacces_image1)
    im_compare.save(chemin_dacces_image2)
    
    im_original = cv2.imread(chemin_dacces_image1)
    im_compare = cv2.imread(chemin_dacces_image2)
    
    im_original_grey = cv2.cvtColor(im_original, cv2.COLOR_RGB2GRAY)
    im_compare_grey = cv2.cvtColor(im_compare, cv2.COLOR_RGB2GRAY)
    
    (ssim_moy, diff) = structural_similarity(im_original_grey, im_compare_grey, full=True) 
    im_diff = (diff*255).astype('uint8')
    chemin_sauv_diff = os.path.join(app.config['DOSSIER_UPLOAD'], 'Image_de_difference.png') 
    plt.imsave(chemin_sauv_diff,im_diff, cmap = 'grey')
    
    im_de_seuil = cv2.threshold(im_diff,0,255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    chemin_sauv_seuil = os.path.join(app.config['DOSSIER_UPLOAD'], 'Image_de_seuil.png') 
    plt.imsave(chemin_sauv_seuil,im_de_seuil)
    
    calcul_contours = cv2.findContours(im_de_seuil.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    calcul_contours = imutils.grab_contours(calcul_contours)
    for contour in calcul_contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        R = np.random.randint(0,255)
        G = np.random.randint(0,255)
        B = np.random.randint(0,255)
        cv2.rectangle(im_original, (x,y), (x+w,y+h),(R,G,B), 2)
        cv2.rectangle(im_compare, (x,y), (x+w,y+h),(R,G,B), 2)
        
    chemin1 = os.path.join(app.config['DOSSIER_UPLOAD'], 'Image_de_ref_contours.png')
    chemin2 = os.path.join(app.config['DOSSIER_UPLOAD'], 'Image_compare_contours.png')
    plt.imsave(chemin1, im_original)  
    plt.imsave(chemin2, im_compare)  
      
    return render_template('traitement_et_calcul.html', ssim_moy = ssim_moy)

if __name__ == '__main__':
    app.run(debug=True)
