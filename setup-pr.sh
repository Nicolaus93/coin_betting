case "$OSTYPE" in
  linux*)   python3 style.py;python3 Data/scripts/generate_optim_config.py;;
  darwin*)  python3 style.py;python3 Data/scripts/generate_optim_config.py;;
  win*)     python style.py;python Data/scripts/generate_optim_config.py;;
  msys*)    python style.py;python Data/scripts/generate_optim_config.py;;
  cygwin*)  python style.py;python Data/scripts/generate_optim_config.py;;
  bsd*)     echo "BSD" ;;
  solaris*) echo "Solaris" ;;
  *)        echo "unknown: $OSTYPE" ;;
esac