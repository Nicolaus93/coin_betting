if __name__ == "__main__":
    import torch
    import argparse
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    from optimal_pytorch.optim import Adam, SGD, SGDOL

    def load_args():
        parser = argparse.ArgumentParser(
            description='Optimizing a quadratic function')

        parser.add_argument('--num-steps',
                            type=int,
                            default=100,
                            help='Number of optimization steps (default: 100)')
        parser.add_argument(
            '--noise-std',
            type=float,
            default=0.1,
            help='The STD of the additive white Gaussian noise (default: 0.1)')
        parser.add_argument('--optim-method',
                            type=str,
                            default='SGD',
                            choices=['Adam', 'SGD', 'SGDOL'],
                            help='which optimizer to use')
        parser.add_argument(
            '--lr',
            type=float,
            default=0.01,
            help='(Except SGDOL) Initial learning rate (default: 0.01)')
        parser.add_argument(
            '--weight-decay',
            type=float,
            default=0.0005,
            help='(All) Weight-decay (L2 penalty) (default: 0.0005)')
        parser.add_argument(
            '--nesterov',
            action='store_true',
            help='(SGD) use nesterov momentum (default: False)')
        parser.add_argument(
            '--momentum',
            type=float,
            default=0.9,
            help='(SGD) Momentum used in optimizer (default: 0.9)')
        parser.add_argument('--beta1',
                            type=float,
                            default=0.9,
                            help='(Adam) beta_1 (default: 0.9)')
        parser.add_argument('--beta2',
                            type=float,
                            default=0.999,
                            help='(Adam) beta_2 (default: 0.999)')
        parser.add_argument('--smoothness',
                            type=float,
                            default=10,
                            help='(SGDOL) smoothness (default: 10)')
        parser.add_argument('--alpha',
                            type=float,
                            default=10,
                            help='(SGDOL) alpha (default: 0.999)')

        return parser.parse_args()

    def main():
        args = load_args()

        # Set up CUDA if needed.
        USE_CUDA = False
        use_cuda = USE_CUDA and torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        # Set the ramdom seed for reproducibility.
        REPRODUCIBLE = True
        SEED = 0
        if REPRODUCIBLE:
            torch.manual_seed(SEED)
            if device != torch.device("cpu"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        # The dimension of the quadratic problem.
        DIMENSION = 10

        # Initialize optimizable parameters.
        x = torch.randn((DIMENSION, 1), requires_grad=True)
        x.to(device)

        # Select optimizer.
        if args.optim_method == 'SGD':
            optimizer = SGD(params=[x],
                            lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay,
                            nesterov=args.nesterov)
        elif args.optim_method == 'Adam':
            optimizer = Adam(params=[x],
                             lr=args.lr,
                             betas=(args.beta1, args.beta2),
                             weight_decay=args.weight_decay)
        elif args.optim_method == 'SGDOL':
            optimizer = SGDOL(params=[x],
                              smoothness=args.smoothness,
                              alpha=args.alpha,
                              weight_decay=args.weight_decay)
        else:
            raise ValueError("Invalid optimizer: {}".format(args.optim_method))

        # Define the objective function y = x^T M x, where M is a positive
        # semi-definite matrix with large condition number.
        M = torch.tensor([
            9.0050, -5.1853, -2.6774, -0.6063, -4.5736, 1.7690, 5.1730,
            -0.7051, 3.5950, -5.5852, -5.1853, 16.0687, -1.7444, 1.6228,
            4.9226, -6.6171, -3.3201, 6.3262, 8.4802, 11.5141, -2.6774,
            -1.7444, 9.7646, 4.1116, 3.3880, 0.7299, 1.5001, -1.3030, -8.5188,
            -5.1559, -0.6063, 1.6228, 4.1116, 16.6046, 1.1656, -1.8692,
            -0.4318, -5.6672, 0.8409, -2.5752, -4.5736, 4.9226, 3.3880, 1.1656,
            14.3299, -3.7571, -2.5134, 2.6684, -4.0715, 11.4973, 1.7690,
            -6.6171, 0.7299, -1.8692, -3.7571, 9.7356, 3.0703, -2.8832,
            -2.8462, -6.1807, 5.1730, -3.3201, 1.5001, -0.4318, -2.5134,
            3.0703, 6.4093, -0.8201, 0.9782, -7.6011, -0.7051, 6.3262, -1.3030,
            -5.6672, 2.6684, -2.8832, -0.8201, 8.1158, 4.3013, 7.6151, 3.5950,
            8.4802, -8.5188, 0.8409, -4.0715, -2.8462, 0.9782, 4.3013, 19.3981,
            8.3670, -5.5852, 11.5141, -5.1559, -2.5752, 11.4973, -6.1807,
            -7.6011, 7.6151, 8.3670, 25.8494
        ])
        M = M.reshape(DIMENSION, DIMENSION)

        def objective(x, M):
            return torch.matmul(torch.matmul(x.t(), M), x)

        # Optimizing the objective function.
        all_obj_values = []
        for _ in tqdm(range(args.num_steps)):
            num_grads = 1 if args.optim_method != 'SGDOL' else 2
            for _ in range(num_grads):
                optimizer.zero_grad()
                obj_value = objective(x, M)
                obj_value.backward()
                x.grad += torch.randn((DIMENSION, 1)) * args.noise_std
                optimizer.step()

            all_obj_values.append(obj_value.item())

        # Draw the training curve.
        linewidth = 3
        fontsize = 14
        plt.figure()
        plt.plot(all_obj_values, linewidth=linewidth)
        plt.xlabel('Iteration', fontsize=fontsize)
        plt.ylabel('Sub-optimality gap', fontsize=fontsize)
        plt.show()

    main()
