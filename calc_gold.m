function seq = calc_gold(c_init, len)
    Nc = 1600;
    n = 2*len + 1 + 1600;
    x1 = zeros(n,1);
    x1(1) = 1;
    x2 = zeros(n,1);
    for i = 0:29
        x2(i+1) = mod((bitshift(c_init, -i)), 2);
    end
    for i = 1:(n-31)
        x1(i+31) = mod((x1(i+3) + x1(i)), 2);
        x2(i+31) = mod((x2(i+3) + x2(i+2) + x2(i+1) + x2(i)), 2);
    end
    seq = zeros(2*len+1,1);
    for n = 1:length(seq)
        seq(n) = mod((x1(n+Nc) + x2(n+Nc)), 2); 
    end
end