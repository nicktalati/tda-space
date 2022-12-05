from cloud import *



if __name__ == "__main__":
    SUBCLOUD_RADIUS = 0.3 # radius of samples taken from larger space
    N_SUBCLOUDS = 10000 # number of samples taken
    HOM_DIM = 1 # homological dimension for which to calculate persistence
    SIGMA = 0.005 # size of gaussian filter that smooths persistence images
    N_BINS = 25 # persistence images are of size N_BINS x N_BINS
    WEIGHT_FUNC = lambda x: x ** 2.5 # scales persistence image based on 
                                     # x = death - birth


    shape = make_torus(1, 0.9, noise=0.02, resolution=80)
    shape = PointCloud(shape)
    shape.visualize()


    colors_list = []
    shape.make_subclouds(SUBCLOUD_RADIUS, n_samples=N_SUBCLOUDS)
        
    kvs = shape.subclouds.items()
    centers = [kv[0][0] for kv in kvs]
    centers = np.asarray(centers)
    subclouds = [np.asarray(kv[1]) for kv in kvs]

    vrp_transformer = VRP(homology_dimensions=[HOM_DIM])
    persistence_diagrams = vrp_transformer.fit_transform(subclouds)
    for sigma in np.linspace(0.0028187920322147652, 0.0029530202194630872, 150):
        print("Sigma = " + str(sigma))
        pi_transformer = PersistenceImage(sigma=sigma,
                                            n_bins=N_BINS,
                                            weight_function=lambda x: x ** 3)
        persistence_images = pi_transformer.fit_transform(persistence_diagrams)

        pi_shape = persistence_images.shape
        images_linearized = persistence_images.reshape(*pi_shape[:2], -1)
        images_linearized = np.squeeze(images_linearized, axis=1)

        pca_transformer = PCA(3, random_state=42)
        colors = pca_transformer.fit_transform(images_linearized)
        colors = normalize(colors)
        colors_list.append(colors)

    colors_list.extend(colors_list[::-1])

    shape = centers

    shape = rotate(shape, 0, 1.6)
    shape = rotate(shape, 2, 0.3)
    pc = otd.geometry.PointCloud()
    pc.points = otd.utility.Vector3dVector(shape)
    pc.colors = otd.utility.Vector3dVector(np.ones_like(shape))

    vis = otd.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pc)

    vis.get_render_option().background_color = \
        np.asarray([28/255, 34/255, 27/255])


    for i in range(300):
        pc.points = otd.utility.Vector3dVector(shape)
        pc.colors = otd.utility.Vector3dVector(colors_list[i])
        shape = rotate(shape, 0, -0.3)
        shape = rotate(shape, 1, 2*np.pi/300)
        shape = rotate(shape, 0, 0.3)
        vis.update_geometry(pc)
        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(True)
        try:
            plt.imsave("color-changing" + "/" + "{:02d}".format(i) + ".png",
            np.asarray(image), dpi=1)
        except OSError:
            os.makedirs("color-changing")
            plt.imsave("color-changing" + "/" + "{:02d}".format(i) + ".png",
            np.asarray(image), dpi=1)

    vis.destroy_window()
    
    """

    t1 = time()

    i = 0

    for w in np.linspace(1.5, 3.75, 40):
        cc = shape.color_cloud(SUBCLOUD_RADIUS,
                            N_SUBCLOUDS,
                            HOM_DIM,
                            sigma=SIGMA,
                            n_bins=N_BINS,
                            weight_function=lambda x: x ** w)

        cc.visualize(save_folder="pngs/" + str(i))
        i += 1

    t2 = time()
    print("Total time:", t2 - t1)
    """
        
    

