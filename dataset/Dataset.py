def Get_DataProvider(args):
    if args.Model_name == 'KELLER':
        from .KELLER import Data_Provider
        return Data_Provider(args)
    else:
        raise NotImplementedError