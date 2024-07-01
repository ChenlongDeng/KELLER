def Get_Model(args):
    if args.Model_name == 'KELLER':
        from .KELLER import KELLER
        return KELLER(args)
    else:
        raise NotImplementedError