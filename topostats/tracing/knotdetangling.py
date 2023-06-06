class nodeStats:
    """Class containing methods to find and analyse the nodes/crossings within a grain"""

    def __init__(self, image: np.ndarray, grains: np.ndarray, skeletons: np.ndarray, px_2_nm: float) -> None:
        self.image = image
        self.grains = grains
        self.skeletons = skeletons
        self.px_2_nm = px_2_nm
        
        """
        a = np.zeros((100,100))
        a[21:80, 20] = 1
        a[21:80, 50] = 1
        a[21:80, 80] = 1
        a[20, 21:80] = 1
        a[80, 21:80] = 1
        a[20, 50] = 0
        a[80, 50] = 0
        self.grains = ndimage.binary_dilation(a, iterations=3)
        self.image = np.ones((100,100))
        self.skeletons = a
        self.px_2_nm = 1
        """

        self.skeleton = None
        self.conv_skelly = None
        self.connected_nodes = None
        self.all_connected_nodes = self.skeletons.copy()

        self.node_centre_mask = None
        self.node_dict = {}
        self.test = None
        self.test2 = None
        self.test3 = None
        self.test4 = None
        self.test5 = None
        self.full_dict = {}
        self.mol_coords = {}
        self.visuals = {}
        self.all_visuals_img = None

    def get_node_stats(self) -> dict:
        """The workflow for obtaining the node statistics.

        Returns:
        --------
        dict
            Key structure:  <grain_number>
                            |-> <node_number>
                                |-> 'error'
                                |-> 'branch_stats'
                                    |-> <branch_number>
                                        |-> 'ordered_coords', 'heights', 'gaussian_fit', 'fwhm', 'angles'
                                |-> 'node stats'
                                    |-> 'node_area_grain', 'node_area_image', 'node_branch_mask', 'node_mid_coords'
        """
        labelled_skeletons = label(self.skeletons)
        for skeleton_no in range(1, labelled_skeletons.max() + 1):
            LOGGER.info(f"Processing Grain: {skeleton_no}")
            self.skeleton = self.skeletons.copy()
            self.skeleton[labelled_skeletons != skeleton_no] = 0
            self.conv_skelly = convolve_skelly(self.skeleton)
            if len(self.conv_skelly[self.conv_skelly == 3]) != 0:  # check if any nodes
                self.connect_close_nodes(node_width=6)
                self.highlight_node_centres(self.connected_nodes)
                self.analyse_nodes(box_length=20)
                if self.check_node_errorless():
                    self.mol_coords[skeleton_no], self.visuals[skeleton_no] = self.compile_trace()
                    pass
                self.full_dict[skeleton_no] = self.node_dict
            else:
                self.full_dict[skeleton_no] = {}
        self.all_visuals_img = dnaTrace.concat_images_in_dict(self.image.shape, self.visuals)

    def check_node_errorless(self):
        for _, vals in self.node_dict.items():
            if vals['error']:
                return False
            else:
                pass
        return True

    def connect_close_nodes(self, node_width: float = 2.85) -> None:
        """Looks to see if nodes are within the node_width boundary (2.85nm) and thus
        are also labeled as part of the node.

        Parameters
        ----------
        node_width: float
            The width of the dna in the grain, used to connect close nodes.
        """
        self.connected_nodes = self.conv_skelly.copy()
        nodeless = self.conv_skelly.copy()
        nodeless[(nodeless == 3) | (nodeless == 2)] = 0  # remove node & termini points
        nodeless_labels = label(nodeless)
        for i in range(1, nodeless_labels.max() + 1):
            if nodeless[nodeless_labels == i].size < (node_width / self.px_2_nm):
                # maybe also need to select based on height? and also ensure small branches classified
                self.connected_nodes[nodeless_labels == i] = 3

    def highlight_node_centres(self, mask):
        """Uses the provided mask to calculate the node centres based on
        height. These node centres are then re-plotted on the mask.

            bg = 0, skeleton = 1, endpoints = 2, node_centres = 3.
        """
        small_node_mask = mask.copy()
        small_node_mask[mask == 3] = 1  # remap nodes to skeleton
        big_nodes = mask.copy()
        big_nodes[mask != 3] = 0  # remove non-nodes
        big_nodes[mask == 3] = 1  # set nodes to 1
        big_node_mask = label(big_nodes)

        for i in np.delete(np.unique(big_node_mask), 0):  # get node indecies
            centre = np.unravel_index((self.image * (big_node_mask == i).astype(int)).argmax(), self.image.shape)
            small_node_mask[centre] = 3

        self.node_centre_mask = small_node_mask

    def analyse_nodes(self, box_length: float = 20):
        """This function obtains the main analyses for the nodes of a single molecule. Within a certain box (nm) around the node.

        bg = 0, skeleton = 1, endpoints = 2, nodes = 3.

        Parameters:
        -----------
        box_length: float
            The side length of the box around the node to analyse (in nm).

        """
        # santity check for box length (too small can cause empty sequence error)
        length = int((box_length / 2) / self.px_2_nm)
        if length < 10:
            LOGGER.info(f"Readapted Box Length from {box_length/2}nm or {length}px to 10px")
            length = 10
        x_arr, y_arr = np.where(self.node_centre_mask.copy() == 3)

        # check whether average trace resides inside the grain mask
        dilate = ndimage.binary_dilation(self.skeleton, iterations=2)
        average_trace_advised = dilate[self.grains == 1].sum() == dilate.sum()
        LOGGER.info(f"Branch height traces will be averaged: {average_trace_advised}")

        # iterate over the nodes to find areas
        #node_dict = {}
        matched_branches = None
        branch_img = None
        avg_img = None

        real_node_count = 0
        for node_no, (x, y) in enumerate(zip(x_arr, y_arr)):  # get centres
            # get area around node - might need to check if box lies on the edge
            image_area = self.image[x - length : x + length + 1, y - length : y + length + 1]
            node_area = self.connected_nodes.copy()[x - length : x + length + 1, y - length : y + length + 1]
            reduced_node_area = self._only_centre_branches(node_area)
            branch_mask = reduced_node_area.copy()
            branch_mask[branch_mask == 3] = 0
            branch_mask[branch_mask == 2] = 1
            node_coords = np.stack(np.where(reduced_node_area == 3)).T
            centre = (np.asarray(node_area.shape) / 2).astype(int)
            error = False  # to see if node too complex or region too small

            # iterate through branches to order
            labeled_area = label(
                branch_mask
            )  # labeling the branch mask may not be the best way to do this due to a pissoble single connection
            LOGGER.info(f"No. branches from node {node_no}: {labeled_area.max()}")

            # for cats paper figures - should be removed
            if node_no == 0:
                self.test = labeled_area
            # stop processing if nib (node has 2 branches)
            if labeled_area.max() <= 2:
                LOGGER.info(f"node {node_no} has only two branches - skipped & nodes removed")
                # sometimes removal of nibs can cause problems when re-indexing nodes
                print(f"{len(node_coords)} pixels in nib node")
                #np.savetxt("/Users/Maxgamill/Desktop/nib.txt", self.node_centre_mask)
                temp = self.node_centre_mask.copy()
                temp_node_coords = node_coords.copy()
                temp_node_coords += ([x, y] - centre)
                temp[temp_node_coords[:,0], temp_node_coords[:,1]] = 1
                #np.savetxt("/Users/Maxgamill/Desktop/nib2.txt", temp)
                # node_coords += ([x, y] - centre) # get whole image coords
                # self.node_centre_mask[x, y] = 1 # remove these from node_centre_mask
                # self.connected_nodes[node_coords[:,0], node_coords[:,1]] = 1 # remove these from connected_nodes
            else:
                try:
                    # check wether resolution good enough to trace
                    res = self.px_2_nm <= 1000 / 512
                    if not res:
                        print("Res Error")
                        raise ResolutionError

                    real_node_count += 1
                    print(f"Real node: {real_node_count}")
                    ordered_branches = []
                    vectors = []
                    for branch_no in range(1, labeled_area.max() + 1):
                        # get image of just branch
                        branch = labeled_area.copy()
                        branch[labeled_area != branch_no] = 0
                        branch[labeled_area == branch_no] = 1
                        # order branch
                        ordered = self.order_branch(branch, centre)
                        #print("ordered: ", ordered)
                        # identify vector
                        vector = self.get_vector(ordered, centre)
                        # add to list
                        vectors.append(vector)
                        ordered_branches.append(ordered)
                    if node_no == 0:
                        self.test2 = vectors
                    # pair vectors
                    pairs = self.pair_vectors(np.asarray(vectors))

                    # join matching branches through node
                    matched_branches = {}
                    branch_img = np.zeros_like(node_area)  # initialising paired branch img
                    avg_img = np.zeros_like(node_area)
                    for i, (branch_1, branch_2) in enumerate(pairs):
                        matched_branches[i] = {}
                        branch_1_coords = ordered_branches[branch_1]
                        branch_2_coords = ordered_branches[branch_2]
                        # find close ends by rearranging branch coords
                        branch_1_coords, branch_2_coords = self.order_branches(branch_1_coords, branch_2_coords)
                        # Linearly interpolate across the node
                        #   binary line needs to consider previous pixel as it can make kinks (non-skelly bits)
                        #   which can cause a pixel to be missed when ordering the traces
                        crossing1 = self.binary_line(branch_1_coords[-1], centre)
                        crossing2 = self.binary_line(centre, branch_2_coords[0])
                        crossing = np.append(crossing1, crossing2).reshape(-1, 2)
                        # remove the duplicate crossing coords
                        uniq_cross_idxs = np.unique(crossing, axis=0, return_index=True)[1]
                        crossing = np.array([crossing[i] for i in sorted(uniq_cross_idxs)])
                        # Branch coords and crossing
                        branch_coords = np.append(branch_1_coords, crossing[1:-1], axis=0)
                        branch_coords = np.append(branch_coords, branch_2_coords, axis=0)
                        # make images of single branch joined and multiple branches joined
                        single_branch = np.zeros_like(node_area)
                        single_branch[branch_coords[:, 0], branch_coords[:, 1]] = 1
                        single_branch = getSkeleton(image_area, single_branch).get_skeleton("zhang")
                        # calc image-wide coords
                        branch_coords_img = branch_coords + ([x, y] - centre)
                        matched_branches[i]["ordered_coords"] = branch_coords_img
                        matched_branches[i]["ordered_coords_local"] = branch_coords
                        # get heights and trace distance of branch
                        distances = self.coord_dist(branch_coords)
                        zero_dist = distances[np.where(np.all(branch_coords == centre, axis=1))]
                        if average_trace_advised:
                            # np.savetxt("knot2/area.txt",image_area)
                            # np.savetxt("knot2/single_branch.txt",single_branch)
                            # print("ZD: ", zero_dist)
                            distances, heights, mask, _ = self.average_height_trace(image_area, single_branch, zero_dist)
                            # add in mid dist adjustment
                            matched_branches[i]["avg_mask"] = mask
                        else:
                            heights = self.image[branch_coords_img[:, 0], branch_coords_img[:, 1]]
                            distances = distances - zero_dist
                        matched_branches[i]["heights"] = heights
                        matched_branches[i]["distances"] = distances

                        # identify over/under
                        fwhm2 = self.fwhm2(heights, distances)
                        matched_branches[i]["fwhm2"] = fwhm2

                    # add paired and unpaired branches to image plot
                    fwhms = []
                    for branch_idx, values in matched_branches.items():
                        fwhms.append(values["fwhm2"][0])
                    branch_idx_order = np.array(list(matched_branches.keys()))[np.argsort(np.array(fwhms))]

                    for i, branch_idx in enumerate(branch_idx_order):
                        branch_coords = matched_branches[branch_idx]["ordered_coords_local"]
                        branch_img[branch_coords[:, 0], branch_coords[:, 1]] = i + 1  # add to branch img
                        if average_trace_advised:  # add avg traces
                            avg_img[matched_branches[branch_idx]["avg_mask"] != 0] = i + 1
                        else:
                            avg_img = None

                    unpaired_branches = np.delete(np.arange(0, labeled_area.max()), pairs.flatten())
                    LOGGER.info(f"Unpaired branches: {unpaired_branches}")
                    branch_label = branch_img.max()
                    for i in unpaired_branches:  # carries on from loop variable i
                        branch_label += 1
                        branch_img[ordered_branches[i][:, 0], ordered_branches[i][:, 1]] = branch_label

                    if node_no == 0:
                        self.test3 = avg_img

                    # calc crossing angle
                    # get full branch vectors
                    vectors = []
                    for branch_no, values in matched_branches.items():
                        vectors.append(self.get_vector(values["ordered_coords"], centre))
                    # calc angles to first vector i.e. first should always be 0
                    cos_angles = self.calc_angles(np.asarray(vectors))[0]
                    cos_angles[cos_angles > 1] = 1  # floating point sometimes causes nans for 1's
                    angles = np.arccos(cos_angles) / np.pi * 180
                    for i, angle in enumerate(angles):
                        matched_branches[i]["angles"] = angle

                    if node_no == 0:
                        self.test4 = vectors
                        self.test5 = angles

                except ValueError:
                    LOGGER.error(f"Node {node_no} too complex, see images for details.")
                    error = True
                except ResolutionError:
                    LOGGER.info(f"Node stats skipped as resolution too low: {self.px_2_nm}nm per pixel")
                    error = True

                print("Error: ", error)
                self.node_dict[real_node_count] = {
                    "error": error,
                    "px_2_nm": self.px_2_nm,
                    "crossing_type": None,
                    "branch_stats": matched_branches,
                    "node_stats": {
                        "node_mid_coords": [x, y],
                        "node_area_image": image_area,
                        "node_area_grain": self.grains[x - length : x + length + 1, y - length : y + length + 1],
                        "node_area_skeleton": node_area,
                        "node_branch_mask": branch_img,
                        "node_avg_mask": avg_img,
                    },
                }

            self.all_connected_nodes[self.connected_nodes != 0] = self.connected_nodes[self.connected_nodes != 0]
            self.node_dict = self.node_dict

    def order_branch(self, binary_image: np.ndarray, anchor: list):
        """Orders a linear branch by identifing an endpoint, and looking at the local area of the point to find the next.

        Parameters
        ----------
        binary_image: np.ndarray
            A binary image of a skeleton segment to order it's points.
        anchor: list
            A list of 2 integers representing the coordinate to order the branch from the endpoint closest to this.

        Returns
        -------
        np.ndarray
            An array of ordered cordinates.
        """
        if len(np.argwhere(binary_image == 1)) < 3: # if < 3 coords just return them
            return np.argwhere(binary_image == 1)
        
        binary_image = np.pad(binary_image, 1).astype(int)
        
        # get branch starts
        endpoints_highlight = ndimage.convolve(binary_image, np.ones((3, 3)))
        endpoints_highlight[binary_image == 0] = 0
        endpoints = np.argwhere(endpoints_highlight == 2)

        if len(endpoints) != 0:
            # as > 1 endpoint, find one closest to anchor
            dist_vals = abs((endpoints - anchor).sum(axis=1))
            start = endpoints[np.argmin(dist_vals)]
        else: # will be circular so pick the first coord (is this always the case?)
            start = np.argwhere(binary_image == 1)[0]

        # add starting point to ordered array
        ordered = []
        ordered.append(start)
        binary_image[start[0], start[1]] = 0  # remove from array

        # iterate to order the rest of the points
        #for i in range(no_points - 1):
        current_point = ordered[-1]  # get last point
        area, _ = self.local_area_sum(binary_image, current_point)  # look at local area
        local_next_point =  np.argwhere(area.reshape((3, 3,)) == 1) - (1, 1)
        while len(local_next_point) != 0:
            next_point = (current_point + local_next_point)[0]
            # find where to go next
            #ordered[i + 1] += next_point  # add to ordered array
            ordered.append(next_point)
            binary_image[next_point[0], next_point[1]] = 0  # set value to zero
            
            current_point = ordered[-1]  # get last point
            area, _ = self.local_area_sum(binary_image, current_point)  # look at local area
            local_next_point =  np.argwhere(area.reshape((3, 3,)) == 1) - (1, 1)

        return np.array(ordered) - [1, 1] # remove padding

    @staticmethod
    def local_area_sum(binary_map, point):
        """Evaluates the local area around a point in a binary map.

        Parameters
        ----------
        binary_map: np.ndarray
            A binary array of an image.
        point: Union[list, touple, np.ndarray]
            A single object containing 2 integers relating to a point within the binary_map

        Returns
        -------
        np.ndarray
            An array values of the local coordinates around the point.
        int
            A value corresponding to the number of neighbours around the point in the binary_map.
        """
        x, y = point
        local_pixels = binary_map[x - 1 : x + 2, y - 1 : y + 2].flatten()
        local_pixels[4] = 0  # ensure centre is 0
        return local_pixels, local_pixels.sum()

    @staticmethod
    def get_vector(coords, origin):
        """Calculate the normalised vector of the coordinate means in a branch"""
        start_coord = coords[np.absolute(origin - coords).sum(axis=1).argmin()]
        vector = coords.mean(axis=0) - start_coord
        vector /= abs(vector).max()
        return vector

    @staticmethod
    def calc_angles(vectors: np.ndarray):
        """Calculates the cosine of the angles between vectors in an array.
        Uses the formula: cos(theta) = |a|•|b|/|a||b|

        Parameters
        ----------
        vectors: np.ndarray
            Array of 2x1 vectors.

        Returns
        -------
        np.ndarray
            An array of the cosine of the angles between the vectors.
        """
        dot = vectors @ vectors.T
        norm = abs(np.diag(dot)) ** 0.5
        angles = abs(dot / (norm.reshape(-1, 1) @ norm.reshape(1, -1)))
        return angles

    def pair_vectors(self, vectors: np.ndarray):
        """Takes a list of vectors and pairs them based on the angle between them

        Parameters
        ----------
        vectors: np.ndarray
            Array of 2x1 vectors to be paired.

        Returns:
        --------
        np.ndarray
            An array of the matching pair indicies.
        """
        # calculate cosine of angle
        angles = self.calc_angles(vectors)
        # find highest values
        np.fill_diagonal(angles, 0)  # ensures not paired with itself
        # match angles
        return self.pair_angles(angles)

    @staticmethod
    def pair_angles(angles):
        """pairs large values in a symmetric NxN matrix"""
        angles_cp = angles.copy()
        pairs = []
        for _ in range(int(angles.shape[0] / 2)):
            pair = np.unravel_index(np.argmax(angles_cp), angles.shape)
            pairs.append(pair)  # add to list
            angles_cp[[pair]] = 0  # set rows 0 to avoid picking again
            angles_cp[:, [pair]] = 0  # set cols 0 to avoid picking again

        return np.asarray(pairs)

    @staticmethod
    def gaussian(x, h, mean, sigma):
        """The gaussian function.

        Parameters
        ----------
        h: float
            The peak height of the gaussian.
        x: np.ndarray
            X values to be passed into the gaussian.
        mean: float
            The mean of the x values.
        sigma: float
            The standard deviation of the image.

        Returns
        -------
        np.ndarray
            The y-values of the gaussian performed on the x values.
        """
        return h * np.exp(-((x - mean) ** 2) / (2 * sigma**2))

    def fwhm(self, heights, distances):
        """Fits a gaussian to the branch heights, and calculates the FWHM"""
        mean = 45.5  # hard coded as middle node value
        sigma = 1 / (200 / 1024)  # 1nm / px2nm = px  half a nm as either side of std
        popt, pcov = optimize.curve_fit(
            self.gaussian,
            distances,
            heights - heights.min(),
            p0=[max(heights) - heights.min(), mean, sigma],
            maxfev=8000,
        )

        return 2.3548 * popt[2], popt  # 2*(2ln2)^1/2 * sigma = FWHM

    def fwhm2(self, heights, distances):
        centre_fraction = int(len(heights) * 0.2)  # incase zone approaches another node, look at centre for max
        high_idx = np.argmax(heights[centre_fraction:-centre_fraction]) + centre_fraction
        heights_norm = heights.copy() - heights.min()  # lower graph so min is 0
        hm = heights_norm.max() / 2  # half max value -> try to make it the same as other crossing branch?

        # get array halves to find first points that cross hm
        arr1 = heights_norm[:high_idx][::-1]
        dist1 = distances[:high_idx][::-1]
        arr2 = heights_norm[high_idx:]
        dist2 = distances[high_idx:]

        arr1_hm = 0
        arr2_hm = 0

        for i in range(len(arr1) - 1):
            if (arr1[i] > hm) and (arr1[i + 1] < hm):  # if points cross through the hm value
                arr1_hm = self.lin_interp([dist1[i], arr1[i]], [dist1[i + 1], arr1[i + 1]], yvalue=hm)
                break

        for i in range(len(arr2) - 1):
            if (arr2[i] > hm) and (arr2[i + 1] < hm):  # if points cross through the hm value
                arr2_hm = self.lin_interp([dist2[i], arr2[i]], [dist2[i + 1], arr2[i + 1]], yvalue=hm)
                break

        fwhm = arr2_hm - arr1_hm

        return fwhm, [arr1_hm, arr2_hm, hm], [high_idx, distances[high_idx], heights[high_idx]]

    @staticmethod
    def lin_interp(point_1, point_2, xvalue=None, yvalue=None):
        """Linear interp 2 points by finding line eq and subbing."""
        m = (point_1[1] - point_2[1]) / (point_1[0] - point_2[0])
        c = point_1[1] - (m * point_1[0])
        if xvalue is not None:
            interp_y = m * xvalue + c
            return interp_y
        if yvalue is not None:
            interp_x = (yvalue - c) / m
            return interp_x

    @staticmethod
    def close_coords(endpoints1, endpoints2):
        """Find the closes coordinates (those at the node crossing) between 2 pairs."""
        sum1 = abs(endpoints1 - endpoints2).sum(axis=1)
        sum2 = abs(endpoints1[::-1] - endpoints2).sum(axis=1)
        if sum1.min() < sum2.min():
            min_idx = np.argmin(sum1)
            return endpoints1[min_idx], endpoints2[min_idx]
        else:
            min_idx = np.argmin(sum2)
            return endpoints1[::-1][min_idx], endpoints2[min_idx]

    @staticmethod
    def order_branches(branch1, branch2):
        """Find the closest coordinates between 2 coordinate arrays."""
        endpoints1 = np.asarray([branch1[0], branch1[-1]])
        endpoints2 = np.asarray([branch2[0], branch2[-1]])
        sum1 = abs(endpoints1 - endpoints2).sum(axis=1)
        sum2 = abs(endpoints1[::-1] - endpoints2).sum(axis=1)
        if sum1.min() < sum2.min():
            if np.argmin(sum1) == 0:
                return branch1[::-1], branch2
            else:
                return branch1, branch2[::-1]
        else:
            if np.argmin(sum2) == 0:
                return branch1, branch2
            else:
                return branch1[::-1], branch2[::-1]

    @staticmethod
    def binary_line(start, end):
        """Creates a binary path following the straight line between 2 points."""
        arr = []
        m_swap = False
        x_swap = False
        slope = (end - start)[1] / (end - start)[0]

        if abs(slope) > 1:  # swap x and y if slope will cause skips
            start, end = start[::-1], end[::-1]
            slope = 1 / slope
            m_swap = True

        if start[0] > end[0]:  # swap x coords if coords wrong way arround
            start, end = end, start
            x_swap = True

        # code assumes slope < 1 hence swap
        x_start, y_start = start
        x_end, y_end = end
        for x in range(x_start, x_end + 1):
            y_true = slope * (x - x_start) + y_start
            y_pixel = np.round(y_true)
            arr.append([x, y_pixel])

        if m_swap:  # if swapped due to slope, return
            arr = np.asarray(arr)[:, [1, 0]].reshape(-1, 2).astype(int)
            if x_swap:
                return arr[::-1]
            else:
                return arr
        else:
            arr = np.asarray(arr).reshape(-1, 2).astype(int)
            if x_swap:
                return arr[::-1]
            else:
                return arr

    @staticmethod
    def coord_dist(coords: np.ndarray, px_2_nm: float = 1) -> np.ndarray:
        """Takes a list/array of coordinates (Nx2) and produces an array which
        accumulates a real distance as if traversing from pixel to pixel.

        Parameters
        ----------
        coords: np.ndarray
            A Nx2 integer array corresponding to the ordered coordinates of a binary trace.
        px_2_nm: float
            The pixel to nanometer scaling factor.

        Returns
        -------
        np.ndarray
            An array of length N containing thcumulative sum of the distances.
        """
        dist_list = [0]
        dist = 0
        for i in range(len(coords) - 1):
            if abs(coords[i] - coords[i + 1]).sum() == 2:
                dist += 2**0.5
            else:
                dist += 1
            dist_list.append(dist)
        return np.asarray(dist_list) * px_2_nm

    @staticmethod
    def above_below_value_idx(array, value):
        """Finds index of the points neighbouring the value in an array."""
        idx1 = abs(array - value).argmin()
        try:
            if value < array[idx1 + 1] and array[idx1] < value:
                idx2 = idx1 + 1
            elif value < array[idx1] and array[idx1 - 1] < value:
                idx2 = idx1 - 1
            else:
                raise IndexError  # this will be if the number is the same
            indices = [idx1, idx2]
            indices.sort()
            return indices
        except IndexError:
            return None

    def average_height_trace(self, img: np.ndarray, branch_mask: np.ndarray, dist_zero_point: float) -> tuple:
        """Dilates the original branch to create two additional side-by-side branches
        in order to get a more accurate average of the height traces. This function produces
        the common distances between these 3 branches, and their averaged heights.

        Parameters
        ----------
        img: np.ndarray
            An array of numbers pertaining to an image.
        branch_mask: np.ndarray
            A binary array of the branch, must share the same dimensions as the image.

        Returns
        -------
        tuple
            A tuple of the averaged heights from the linetrace and their corresponding distances
            from the crossing.
        """
        # get heights and dists of the original (middle) branch
        branch_coords = np.stack(np.where(branch_mask == 1)).T
        branch_dist = self.coord_dist(branch_coords)
        branch_heights = img[branch_coords[:, 0], branch_coords[:, 1]]
        branch_dist_norm = branch_dist - dist_zero_point  # branch_dist[branch_heights.argmax()]

        # want to get a 3 pixel line trace, one on each side of orig
        dilate = ndimage.binary_dilation(branch_mask, iterations=1)
        dilate_minus = dilate.copy()
        dilate_minus[branch_mask == 1] = 0
        dilate2 = ndimage.binary_dilation(dilate)
        dilate2[(dilate == 1) | (branch_mask == 1)] = 0
        labels = label(dilate2)
        # if parallel trace out and back in zone, can get > 2 labels
        labels = self._remove_re_entering_branches(labels, remaining_branches=2)
        # if parallel trace doesn't exit window, can get 1 label
        #   occurs when skeleton has poor connections (extra branches which cut corners)
        if labels.max() == 1:
            conv = convolve_skelly(branch_mask)
            endpoint = np.stack(np.where(conv == 2)).T
            para_trace_coords = np.stack(np.where(labels == 1)).T
            abs_diff = np.absolute(para_trace_coords - endpoint).sum(axis=1)
            min_idxs = np.where(abs_diff == abs_diff.min())
            trace_coords_remove = para_trace_coords[min_idxs]
            labels[trace_coords_remove[:, 0], trace_coords_remove[:, 1]] = 0
            labels = label(labels)

        # reduce binary dilation distance
        paralell = np.zeros_like(branch_mask).astype(np.int32)
        for i in range(1, labels.max() + 1):
            single = labels.copy()
            single[single != i] = 0
            single[single == i] = 1
            sing_dil = ndimage.binary_dilation(single)
            paralell[(sing_dil == dilate_minus) & (sing_dil == 1)] = i
        labels = paralell.copy()
        # print(np.unique(labels, return_index=True))

        binary = labels.copy()
        binary[binary != 0] = 1
        binary += branch_mask

        # get and order coords, then get heights and distances relitive to node centre / highest point
        centre_fraction = 1 - 0.8  # the middle % of data to look for peak - stops peaks being found at edges
        heights = []
        distances = []
        for i in range(1, labels.max() + 1):
            trace = labels.copy()
            trace[trace != i] = 0
            trace[trace != 0] = 1
            trace = getSkeleton(img, trace).get_skeleton("zhang")
            trace = self.order_branch(trace, branch_coords[0])
            height_trace = img[trace[:, 0], trace[:, 1]]
            height_len = len(height_trace)
            central_heights = height_trace[int(height_len * centre_fraction) : int(-height_len * centre_fraction)]
            heights.append(height_trace)
            dist = self.coord_dist(trace)
            distances.append(
                dist - dist_zero_point
            )  # branch_dist[branch_heights.argmax()]) #dist[central_heights.argmax()])

        # Make like coord system using original branch
        avg1 = []
        avg2 = []
        for mid_dist in branch_dist_norm:
            for i, (distance, height) in enumerate(zip(distances, heights)):
                # check if distance already in traces array
                if (mid_dist == distance).any():
                    idx = np.where(mid_dist == distance)
                    if i == 0:
                        avg1.append([mid_dist, height[idx][0]])
                    else:
                        avg2.append([mid_dist, height[idx][0]])
                # if not, linearly interpolate the mid-branch value
                else:
                    # get index after and before the mid branches' x coord
                    xidxs = self.above_below_value_idx(distance, mid_dist)
                    if xidxs is None:
                        pass  # if indexes outside of range, pass
                    else:
                        point1 = [distance[xidxs[0]], height[xidxs[0]]]
                        point2 = [distance[xidxs[1]], height[xidxs[1]]]
                        y = self.lin_interp(point1, point2, xvalue=mid_dist)
                        if i == 0:
                            avg1.append([mid_dist, y])
                        else:
                            avg2.append([mid_dist, y])
        avg1 = np.asarray(avg1)
        avg2 = np.asarray(avg2)

        # ensure arrays are same length to average
        temp_x = branch_dist_norm[np.isin(branch_dist_norm, avg1[:, 0])]
        common_dists = avg2[:, 0][np.isin(avg2[:, 0], temp_x)]

        common_avg_branch_heights = branch_heights[np.isin(branch_dist_norm, common_dists)]
        common_avg1_heights = avg1[:, 1][np.isin(avg1[:, 0], common_dists)]
        common_avg2_heights = avg2[:, 1][np.isin(avg2[:, 0], common_dists)]

        average_heights = (common_avg_branch_heights + common_avg1_heights + common_avg2_heights) / 3
        return (
            common_dists,
            average_heights,
            binary,
            [[heights[0], branch_heights, heights[1]], [distances[0], branch_dist_norm, distances[1]]],
        )

    @staticmethod
    def _remove_re_entering_branches(image: np.ndarray, remaining_branches: int = 1) -> np.ndarray:
        """Looks to see if branches exit and re-enter the viewing area, then removes one-by-one
        the smallest, so that only <remaining_branches> remain.
        """
        rtn_image = image.copy()
        binary_image = image.copy()
        binary_image[binary_image != 0] = 1
        labels = label(binary_image)

        if labels.max() > remaining_branches:
            lens = [labels[labels == i].size for i in range(1, labels.max() + 1)]
            while len(lens) > remaining_branches:
                smallest_idx = min(enumerate(lens), key=lambda x: x[1])[0]
                rtn_image[labels == smallest_idx + 1] = 0
                lens.remove(min(lens))

        return rtn_image

    @staticmethod
    def _only_centre_branches(node_image: np.ndarray):
        """Looks identifies the node being examined and removes all
        branches not connected to it.

        Parameters
        ----------
        node_image : np.ndarray
            An image of the skeletonised area surrounding the node where
            the background = 0, skeleton = 1, termini = 2, nodes = 3.

        Returns
        -------
        np.ndarray
            The initial node image but only with skeletal branches
            connected to the middle node.
        """
        node_image_cp = node_image.copy()

        # get node-only image
        nodes = node_image_cp.copy()
        nodes[nodes != 3] = 0
        labeled_nodes = label(nodes)

        # find which cluster is closest to the centre
        centre = np.asarray(node_image_cp.shape) / 2
        node_coords = np.stack(np.where(nodes == 3)).T
        min_coords = node_coords[abs(node_coords - centre).sum(axis=1).argmin()]
        centre_idx = labeled_nodes[min_coords[0], min_coords[1]]

        # get nodeless image
        nodeless = node_image_cp.copy()
        nodeless[nodeless == 3] = 0
        nodeless[nodeless == 2] = 1  # if termini, need this in the labeled branches too
        nodeless[labeled_nodes == centre_idx] = 1  # return centre node
        labeled_nodeless = label(nodeless)

        # apply to return image
        for i in range(1, labeled_nodeless.max() + 1):
            if (node_image_cp[labeled_nodeless == i] == 3).any():
                node_image_cp[labeled_nodeless != i] = 0
                break

        return node_image_cp

    def compile_trace(self):
        """This function uses the branches and FWHM's identified in the node_stats dictionary to create a
        continious trace of the molecule.
        """
        LOGGER.info("Compiling the trace.")

        # iterate throught the dict to get branch coords, heights and fwhms
        node_centre_coords = []
        node_area_box = []
        crossing_coords = []
        crossing_heights = []
        crossing_distances = []
        fwhms = []
        for _, stats in self.node_dict.items():
            node_centre_coords.append(stats['node_stats']['node_mid_coords'])
            node_area_box.append(stats['node_stats']['node_area_image'].shape)
            temp_coords = []
            temp__heights = []
            temp_distances = []
            temp_fwhms = []
            for _, branch_stats in stats['branch_stats'].items():
                temp_coords.append(branch_stats['ordered_coords'])
                temp__heights.append(branch_stats['heights'])
                temp_distances.append(branch_stats['distances'])
                temp_fwhms.append(branch_stats["fwhm2"][0])
            crossing_coords.append(temp_coords)
            crossing_heights.append(temp__heights)
            crossing_distances.append(temp_distances)
            fwhms.append(temp_fwhms)

        #print(fwhms)

        # get image minus the crossing areas
        minus = self.get_minus_img(node_area_box, node_centre_coords)
        # get crossing image
        crossings = self.get_crossing_img(crossing_coords, minus.max() + 1)
        #print(crossing_coords)
        # combine branches and segments
        both_img = self.get_both_img(minus, crossings)

        #np.savetxt("/Users/Maxgamill/Desktop/minus.txt", minus)
        #np.savetxt("/Users/Maxgamill/Desktop/cross.txt", crossings)
        #np.savetxt("/Users/Maxgamill/Desktop/both.txt", both_img)
        #np.savetxt("/Users/Maxgamill/Desktop/skel.txt", self.skeleton)

        # order minus segments
        ordered = []
        for i in range(1, minus.max() + 1):
            arr = np.where(minus, minus == i, 0)
            #np.savetxt("/Users/Maxgamill/Desktop/arr.txt", arr)
            ordered.append(self.order_branch(arr, [0, 0]))  # orientated later

        # combine ordered indexes
        for node_crossing_coords in crossing_coords:
            for single_cross in node_crossing_coords:
                ordered.append(np.array(single_cross))
        print("LEN: ", len(ordered))

        print("Getting coord trace")
        coord_trace = self.trace_mol(ordered, both_img)

        for trace in coord_trace:
            print("DISTANCE: ", self.coord_dist(trace, self.px_2_nm)[-1])

        #np.savetxt("/Users/Maxgamill/Desktop/trace.txt", coord_trace[0])

        # visual over under img
        visual = self.get_visual_img(coord_trace, fwhms, crossing_coords)

        #np.savetxt("/Users/Maxgamill/Desktop/visual.txt", visual)

        # I could use the traced coords, remove the node centre coords, and re-label segments
        #   following 1, 2, 3... around the mol which should look like the Planar Diagram formation
        #   (https://topoly.cent.uw.edu.pl/dictionary.html#codes). Then look at each corssing zone again,
        #   determine which in in-undergoing and assign labels counter-clockwise
        print("Getting PD Codes:")
        pd_codes = self.get_pds(coord_trace, node_centre_coords, fwhms, crossing_coords)

        return coord_trace, visual

    def get_minus_img(self, node_area_box, node_centre_coords):
        minus = self.skeleton.copy()
        for i, area in enumerate(node_area_box):
            x, y = node_centre_coords[i]
            area = np.array(area) // 2
            minus[x - area[0] : x + area[0], y - area[1] : y + area[1]] = 0
        return label(minus)

    def get_crossing_img(self, crossing_coords, label_offset):
        crossings = np.zeros_like(self.skeleton)
        i = 0
        for node_crossing_coords in crossing_coords:
            for single_cross_coords in node_crossing_coords:
                crossings[single_cross_coords[:, 0], single_cross_coords[:, 1]] = i + label_offset
                i += 1
        return crossings

    @staticmethod
    def get_both_img(minus_img, crossing_img):
        both_img = minus_img.copy()
        both_img[crossing_img != 0] = crossing_img[crossing_img != 0]
        return both_img

    @staticmethod
    def trace_mol(ordered_segment_coords, both_img):
        remaining = both_img.copy().astype(np.int32)  # image
        # get first segment
        idx = 0  # set index
        coord_trace = ordered_segment_coords[idx].astype(np.int32).copy()  # add ordered segment to trace
        remaining[remaining == idx + 1] = 0  # remove segment from image
        x, y = coord_trace[-1]  # find end coords of trace
        idx = remaining[x - 1 : x + 2, y - 1 : y + 2].max() - 1  # find local area of end coord to find next index

        mol_coords = []
        mol_num = 0
        while len(np.unique(remaining)) > 1:
            mol_num += 1
            while idx > -1:  # either cycled through all or hits terminus -> all will be just background
                if (
                    abs(coord_trace[-1] - ordered_segment_coords[idx][0]).sum()
                    < abs(coord_trace[-1] - ordered_segment_coords[idx][-1]).sum()
                ):
                    coord_trace = np.append(coord_trace, ordered_segment_coords[idx].astype(np.int32), axis=0)
                else:
                    coord_trace = np.append(coord_trace, ordered_segment_coords[idx][::-1].astype(np.int32), axis=0)
                x, y = coord_trace[-1]
                remaining[remaining == idx + 1] = 0
                idx = remaining[x - 1 : x + 2, y - 1 : y + 2].max() - 1  # should only be one value
            mol_coords.append(coord_trace)
            try:
                idx = np.unique(remaining)[1] - 1  # avoid choosing 0
                coord_trace = ordered_segment_coords[idx].astype(np.int32).copy()
            except:  # index of -1 out of range
                break

        print(f"Mols in trace: {len(mol_coords)}")

        return mol_coords
    
    @staticmethod
    def get_trace_idxs(fwhms: list) -> tuple:
        # node fwhms can be a list of different lengths so cannot use np arrays
        under_idxs = []
        over_idxs = []
        for node_fwhms in fwhms:
            order = np.argsort(node_fwhms)
            under_idxs.append(order[0])
            over_idxs.append(order[-1])
        return under_idxs, over_idxs

    def get_visual_img(self, coord_trace, fwhms, crossing_coords):
        # put down traces
        img = np.zeros_like(self.skeleton)
        for mol_no, coords in enumerate(coord_trace):
            temp_img = np.zeros_like(img)
            temp_img[coords[:, 0], coords[:, 1]] = 1
            temp_img = binary_dilation(temp_img)
            img[temp_img != 0] = mol_no + 1

        #np.savetxt("/Users/Maxgamill/Desktop/preimg.txt", img)

        lower_idxs, upper_idxs = self.get_trace_idxs(fwhms)

        if len(coord_trace) > 1:
            for type_idxs in [lower_idxs, upper_idxs]:
                for (node_crossing_coords, type_idx) in zip(crossing_coords, type_idxs):
                    temp_img = np.zeros_like(img)
                    cross_coords = node_crossing_coords[type_idx]
                    # decide which val
                    matching_coords = np.array([])
                    for trace in coord_trace:
                        c = 0
                        # get overlaps between segment coords and crossing under coords
                        for cross_coord in cross_coords:
                            c += ((trace == cross_coord).sum(axis=1) == 2).sum()
                        matching_coords = np.append(matching_coords, c)
                    val = matching_coords.argmax() + 1
                    temp_img[cross_coords[:, 0], cross_coords[:, 1]] = 1
                    temp_img = binary_dilation(temp_img)
                    img[temp_img != 0] = val
        else:
            # make plot where overs are one colour and unders another
            for i, type_idxs in enumerate([lower_idxs, upper_idxs]):
                for (crossing, type_idx) in zip(crossing_coords, type_idxs):
                    temp_img = np.zeros_like(img)
                    cross_coords = crossing[type_idx]
                    # decide which val
                    matching_coords = np.array([])
                    c = 0
                    # get overlaps between segment coords and crossing under coords
                    for cross_coord in cross_coords:
                        c += ((coord_trace[0] == cross_coord).sum(axis=1) == 2).sum()
                    matching_coords = np.append(matching_coords, c)
                    val = matching_coords.argmax() + 1
                    temp_img[cross_coords[:, 0], cross_coords[:, 1]] = 1
                    temp_img = binary_dilation(temp_img)
                    img[temp_img != 0] = i + 2
        return img

    def get_pds(self, trace_coords, node_centres, fwhms, crossing_coords):
        # find idxs of branches from start
        for mol_num, mol_trace in enumerate(trace_coords):
            print(f"Molecule {mol_num}")
            node_coord_idxs = np.array([]).astype(np.int32)
            global_node_idxs = np.array([]).astype(np.int32)
            img = np.zeros_like(self.skeleton.copy()).astype(np.int32)
            for i, c in enumerate(node_centres):
                node_coord_idx = np.where((mol_trace[:, 0] == c[0]) & (mol_trace[:, 1] == c[1]))
                node_coord_idxs = np.append(node_coord_idxs, node_coord_idx)
                global_node_idx = np.zeros_like(node_coord_idx) + i
                global_node_idxs = np.append(global_node_idxs, global_node_idx)

            ordered_node_coord_idxs, ordered_node_idx_idxs = np.sort(node_coord_idxs), np.argsort(node_coord_idxs)
            global_node_idxs = global_node_idxs[ordered_node_idx_idxs]

            # break out from trace loop if "molecule" trace segment contains no nodes
            if len(node_coord_idxs) == 0:
                print("Segment doesn't cross nodes")
            else:
                under_branch_idxs, _ = self.get_trace_idxs(fwhms)
                # iterate though nodes and label segments to node
                img[mol_trace[0 : ordered_node_coord_idxs[0], 0], mol_trace[0 : ordered_node_coord_idxs[0], 1]] = 1
                print("ORDERED: ", ordered_node_coord_idxs)
                for i in range(0, len(ordered_node_coord_idxs) - 1):
                    img[
                        mol_trace[ordered_node_coord_idxs[i] : ordered_node_coord_idxs[i + 1], 0],
                        mol_trace[ordered_node_coord_idxs[i] : ordered_node_coord_idxs[i + 1], 1],
                    ] = (
                        i + 2
                    )
                if sum(abs(mol_trace[0]-mol_trace[-1])) <= 2: # check if mol circular via start and end dist - should probs do root(2)
                    j = 1 # rejoins start at 1
                else:
                    j = i + 1 # doesn't rejoin start
                img[
                    mol_trace[ordered_node_coord_idxs[-1] : -1, 0], mol_trace[ordered_node_coord_idxs[-1] : -1, 1]
                ] = j  
                
                #np.savetxt("/Users/Maxgamill/Desktop/smth.txt", img)
                
                # want to generate PD code by looking at each node and decide which
                #   img label is the under-in one, then append anti-clockwise labels
                #   - We'll have to match the node number to the node order
                #   - Then check the FWHMs to see lowest
                #   - Use lowest FWHM index to get the under branch coords
                #   - Count overlapping coords between under branch coords and each ordered segment
                #   - Get img label of two highest count (in and out)
                #   - Under-in = lowest of two indexes ?? (back 2 start?)

                #print("global node idxs", global_node_idxs)
                pd_code = ''
                for i, global_node_idx in enumerate(global_node_idxs):
                    #print(f"\n----Trace Node Num: {i+1}, Global Node Num: {global_node_idx}----")
                    under_branch_idx = under_branch_idxs[global_node_idx]
                    #print("under_branch_idx: ", under_branch_idx)
                    matching_coords = np.array([])
                    x, y = node_centres[global_node_idx]
                    node_area = img[x - 3 : x + 4, y - 3 : y + 4]
                    uniq_labels = np.unique(node_area)
                    uniq_labels = uniq_labels[uniq_labels != 0]
                    print("uniq labels: ", uniq_labels)
                    #np.savetxt("/Users/Maxgamill/Desktop/na.txt", node_area)

                    for label2 in uniq_labels:
                        c = 0
                        # get overlaps between segment coords and crossing under coords
                        for ordered_branch_coord in crossing_coords[global_node_idx][
                            under_branch_idx
                        ]:  # for global_node[4] branch index is incorrect
                            c += ((np.stack(np.where(img == label2)).T == ordered_branch_coord).sum(axis=1) == 2).sum()
                        matching_coords = np.append(matching_coords, c)
                        #print(f"Segment: {label2.max()}, Matches: {c}")
                    highest_count_labels = [uniq_labels[i] for i in np.argsort(matching_coords)[-2:]]
                    #print("highest count: ", highest_count_labels)
                    if abs(highest_count_labels[0] - highest_count_labels[1]) > 1: # assumes matched branch
                        under_in = max(highest_count_labels)
                    else:
                        under_in = min(highest_count_labels)  # under-in for global_node[4] is incorrect
                    # print(f"Under-in: {under_in}")
                    anti_clock = list(self.vals_anticlock(node_area, under_in))

                    if len(anti_clock) == 2: # mol passes over/under another mol (maybe && [i]+1 == [i+1])
                        pd = f"V{anti_clock};"
                        self.node_dict[global_node_idx+1]["crossing_type"] = "passive"
                    elif len(anti_clock) == 3: # trival crossing (maybe also applies to Y's therefore maybe && consec when sorted)
                        pd = f"Y{anti_clock};"
                        self.node_dict[global_node_idx+1]["crossing_type"] = "trivial"
                    else:
                        pd = f"X{anti_clock};"
                        self.node_dict[global_node_idx+1]["crossing_type"] = "real"
                    print(f"Crossing PD: {pd}")
                    pd_code += pd
                
                print(f"Total PD code: {pd_code}")
                try:
                    topology = None #homfly(pd_code, closure=params.Closure.CLOSED, chiral = True) Need to fix cat pd codes first
                    print(f"Topology: {topology}")
                except:
                    topology = None
                    print("Topology undetermined")
                self.node_dict["topology"] = topology
                

        return None

    @staticmethod
    def make_arr_consec(arr):
        for i, val in enumerate(arr):
            if i not in arr:
                arr[arr >= i] += -1
        return arr

    @staticmethod
    def vals_anticlock(area, start_lbl):
        """Gets the first occurance of values around the edges of an array in an anti-clockwise direction from the start point.

        Parameters
        ----------
        area : np.ndarray
            The labeled image array you want to observe around
        start_lbl : int
            The value to start the anti-clockwise labeling from. Must be an value on the edge of the area array.

        Returns
        -------
        np.ndarray
            An array of the labeled area values in an anti-clockwise direction from the startpoint.
        """
        top = area[0, :][area[0, :] != 0][::-1]
        left = area[:, 0][area[:, 0] != 0]
        bottom = area[-1, :][area[-1, :] != 0]
        right = area[:-1][area[:-1] != 0][::-1]
        total = np.concatenate([top, left, bottom, right])

        # prevent multiple occurances while retaining order
        uniq_total_idxs = np.unique(total, return_index=True)[1]
        total = np.array([total[i] for i in sorted(uniq_total_idxs)])
        start_idx = np.where(total == start_lbl)[0]

        return np.roll(total, -start_idx)