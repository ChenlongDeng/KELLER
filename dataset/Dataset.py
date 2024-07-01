def Get_DataProvider(args):
    if args.Model_name in ['BERT', 'RoBERTa', 'BGE', 'SAILER', 'Lawformer']:
        from .DualEncoder import Data_Provider
        return Data_Provider(args)
    elif args.Model_name == 'BERTPLI':
        from .BERTPLI import Data_Provider
        return Data_Provider(args)
    elif args.Model_name == 'DistillReasoning':
        from .DistillReasoning import Data_Provider
        return Data_Provider(args)
    elif args.Model_name == 'FactReasoning_Concat':
        from .DistillReasoning import Data_Provider
        return Data_Provider(args)
    elif args.Model_name == 'KELLER':
        from .KELLER import Data_Provider
        return Data_Provider(args)
    else:
        raise NotImplementedError