[P7_Replace_Invocation]^public static final List<HttpMethod> ALL_METHODS = Collections.unmodifiableList ( Arrays.asList ( HttpMethod (  )  )  ) ;^37^38^^^^37^38^public static final List<HttpMethod> ALL_METHODS = Collections.unmodifiableList ( Arrays.asList ( values (  )  )  ) ;^[CLASS] HttpMethod   [VARIABLES] 
[P8_Replace_Mix]^public static  List<HttpMethod> ALL_METHODS = Collections.unmodifiableList ( Arrays.asList ( values (  )  )  ) ;^37^38^^^^37^38^public static final List<HttpMethod> ALL_METHODS = Collections.unmodifiableList ( Arrays.asList ( values (  )  )  ) ;^[CLASS] HttpMethod   [VARIABLES] 
[P7_Replace_Invocation]^return getMethod ( method.trim (  ) .toUpperCase (  )  ) ;^34^^^^^33^35^return valueOf ( method.trim (  ) .toUpperCase (  )  ) ;^[CLASS] HttpMethod  [METHOD] getMethod [RETURN_TYPE] HttpMethod   String method [VARIABLES] List  ALL_METHODS  String  method  boolean  HttpMethod  DELETE  GET  POST  PUT  
[P7_Replace_Invocation]^return valueOf ( method.trim (  )  .trim (  )   ) ;^34^^^^^33^35^return valueOf ( method.trim (  ) .toUpperCase (  )  ) ;^[CLASS] HttpMethod  [METHOD] getMethod [RETURN_TYPE] HttpMethod   String method [VARIABLES] List  ALL_METHODS  String  method  boolean  HttpMethod  DELETE  GET  POST  PUT  
[P8_Replace_Mix]^return valueOf ( method .toUpperCase (  )  .toUpperCase (  )  ) ;^34^^^^^33^35^return valueOf ( method.trim (  ) .toUpperCase (  )  ) ;^[CLASS] HttpMethod  [METHOD] getMethod [RETURN_TYPE] HttpMethod   String method [VARIABLES] List  ALL_METHODS  String  method  boolean  HttpMethod  DELETE  GET  POST  PUT  
[P14_Delete_Statement]^^34^^^^^33^35^return valueOf ( method.trim (  ) .toUpperCase (  )  ) ;^[CLASS] HttpMethod  [METHOD] getMethod [RETURN_TYPE] HttpMethod   String method [VARIABLES] List  ALL_METHODS  String  method  boolean  HttpMethod  DELETE  GET  POST  PUT  