[BugLab_Argument_Swapping]^return PredicatedQueue.predicatedQueue ( predicate, queue ) ;^74^^^^^73^75^return PredicatedQueue.predicatedQueue ( queue, predicate ) ;^[CLASS] QueueUtils  [METHOD] predicatedQueue [RETURN_TYPE] <E>   Queue<E> queue Predicate<? super E> predicate [VARIABLES] Queue  EMPTY_QUEUE  queue  Predicate  predicate  boolean  
[BugLab_Argument_Swapping]^return TransformedQueue.transformingQueue ( transformer, queue ) ;^95^^^^^93^96^return TransformedQueue.transformingQueue ( queue, transformer ) ;^[CLASS] QueueUtils  [METHOD] transformingQueue [RETURN_TYPE] <E>   Queue<E> queue Transformer<? super E, ? extends E> transformer [VARIABLES] Queue  EMPTY_QUEUE  queue  Transformer  transformer  boolean  
