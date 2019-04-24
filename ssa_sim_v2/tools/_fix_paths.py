def fix_paths(is_package=True):
    
    # Add package parent folder path to sys.path to make calling packages possible
    
    import os, sys
    rel_path = os.path.dirname(os.path.abspath(__file__))
    
    while os.path.isfile(os.path.abspath(os.path.join(rel_path, "__init__.py"))):
        rel_path = os.path.join(rel_path, "..")
        
    if not is_package:
        rel_path = os.path.join(rel_path, "..")

    main_path = os.path.abspath(rel_path)
    if main_path in sys.path:
        sys.path.remove(main_path)
    if main_path not in sys.path:
        sys.path.insert(0, main_path)

    os.chdir(os.path.abspath(os.path.join(main_path)))