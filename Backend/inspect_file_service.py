import importlib, traceback
try:
    fs = importlib.import_module('file_service')
    print('MODULE OK')
    print('module file:', getattr(fs, '__file__', None))
    print('has process_uploaded_file:', hasattr(fs, 'process_uploaded_file'))
    print('dir:', [n for n in dir(fs) if not n.startswith('__')])
    # show file content
    try:
        import os
        print('file size:', os.path.getsize(fs.__file__))
        with open(fs.__file__, 'rb') as fh:
            raw = fh.read()
        print('raw bytes length:', len(raw))
        try:
            print(raw.decode('utf-8'))
        except Exception as e:
            print('could not decode bytes:', e)
    except Exception as e:
        print('could not read module file:', e)
except Exception as e:
    print('IMPORT EXCEPTION:')
    traceback.print_exc()
    print('str:', str(e))
