import logging, os


fmt = '%(asctime)s %(filename)s %(lineno)d: %(message)s'
datefmt = '%m-%d %H:%M:%S'


def get_logger(name=None, log_file=None, log_level=logging.INFO):
    """ concise log """
    logger = logging.getLogger(name)
    logging.basicConfig(format=fmt, datefmt=datefmt)
    if log_file is not None:
        log_file_folder = os.path.split(log_file)[0]
        if log_file_folder:
            os.makedirs(log_file_folder, exist_ok=True)
        fh = logging.FileHandler(log_file, 'w', encoding='utf-8')
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(fh)
    logger.setLevel(log_level)
    return logger

logger = get_logger()


def log_df_basic_info(df, comments=''):
    if comments:
        logger.info(f'comments {comments}')
    logger.info(f'df.shape {df.shape}')
    logger.info(f'df.columns {df.columns.to_list()}')
    logger.info(f'df.head()\n{df.head()}')
    logger.info(f'df.tail()\n{df.tail()}')
