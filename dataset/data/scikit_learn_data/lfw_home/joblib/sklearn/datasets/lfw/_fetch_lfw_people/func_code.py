# first line: 201
def _fetch_lfw_people(data_folder_path, slice_=None, color=False, resize=None,
                      min_faces_per_person=0):
    """Perform the actual data loading for the lfw people dataset

    This operation is meant to be cached by a joblib wrapper.
    """
    # scan the data folder content to retain people with more that
    # `min_faces_per_person` face pictures
    person_names, file_paths = [], []
    for person_name in sorted(listdir(data_folder_path)):
        folder_path = join(data_folder_path, person_name)
        if not isdir(folder_path):
            continue
        paths = [join(folder_path, f) for f in sorted(listdir(folder_path))]
        n_pictures = len(paths)
        if n_pictures >= min_faces_per_person:
            person_name = person_name.replace('_', ' ')
            person_names.extend([person_name] * n_pictures)
            file_paths.extend(paths)

    n_faces = len(file_paths)
    if n_faces == 0:
        raise ValueError("min_faces_per_person=%d is too restrictive" %
                         min_faces_per_person)

    target_names = np.unique(person_names)
    target = np.searchsorted(target_names, person_names)

    faces = _load_imgs(file_paths, slice_, color, resize)

    # shuffle the faces with a deterministic RNG scheme to avoid having
    # all faces of the same person in a row, as it would break some
    # cross validation and learning algorithms such as SGD and online
    # k-means that make an IID assumption

    indices = np.arange(n_faces)
    np.random.RandomState(42).shuffle(indices)
    faces, target = faces[indices], target[indices]
    return faces, target, target_names
